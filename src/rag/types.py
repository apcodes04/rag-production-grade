"""Shared data shapes for the whole pipeline.

Why one file: every module speaks the same vocabulary, and a reader can learn
the system by reading these ten classes. Why frozen dataclasses for Page and
Chunk: they are facts about a document, nobody should mutate them halfway
through the pipeline. Candidate is NOT frozen because each retrieval stage
deliberately annotates it with one more score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Page:
    doc_id: str          # sha256 of the file bytes: same content means same id
    source: str          # file name, shown to the user in citations
    page: int | None     # 1 based for PDFs, None for plain text
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str        # f"{doc_id}-{index:05d}", deterministic and unique
    doc_id: str
    source: str
    page: int | None
    chunk_index: int
    text: str
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Chunk:
        return Chunk(**data)


@dataclass
class Candidate:
    """One retrieved chunk plus every score it collected on the way."""
    chunk: Chunk
    dense_score: float | None = None     # cosine similarity, roughly -1 to 1
    sparse_score: float | None = None    # BM25, unbounded positive
    rank_dense: int | None = None        # 1 based rank in the dense list
    rank_sparse: int | None = None
    fused_score: float | None = None     # RRF or weighted fusion
    rerank_score: float | None = None    # cross encoder logit
    final_rank: int | None = None

    @property
    def score(self) -> float:
        """The score the last stage produced: what ordering actually used."""
        for value in (self.rerank_score, self.fused_score, self.dense_score, self.sparse_score):
            if value is not None:
                return value
        return 0.0

    def explain(self, *, text_chars: int = 0) -> dict[str, Any]:
        """Every score this chunk collected, for debugging one retrieval.

        The chunk text is opt in. The API sends it so a reader can check the
        answer against the evidence; evaluation reports leave it out, because
        twenty one questions times three sources of full text turns a report
        anyone can skim into a file nobody opens.
        """
        payload = {
            "chunk_id": self.chunk.chunk_id, "source": self.chunk.source,
            "page": self.chunk.page, "dense": self.dense_score, "sparse": self.sparse_score,
            "rank_dense": self.rank_dense, "rank_sparse": self.rank_sparse,
            "fused": self.fused_score, "rerank": self.rerank_score,
        }
        if text_chars:
            text = self.chunk.text
            payload["text"] = text[:text_chars] + ("..." if len(text) > text_chars else "")
            payload["tokens"] = self.chunk.token_count
        return payload


@dataclass
class RetrievalResult:
    """Candidates plus the honest status of how they were found.

    A plain list cannot say "the embedding API was down, these came from
    keyword search only", and that is exactly the thing nobody must miss.
    """
    candidates: list[Candidate] = field(default_factory=list)
    degraded: bool = False
    degraded_reason: str | None = None


@dataclass
class Turn:
    role: str            # "user" or "assistant"
    content: str


@dataclass
class Usage:
    embed_calls: int = 0
    embed_texts: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0

    thinking_tokens: int = 0     # Gemini 3 reasons before answering, and bills for it

    def add(self, other: Usage) -> Usage:
        return Usage(
            self.embed_calls + other.embed_calls, self.embed_texts + other.embed_texts,
            self.llm_calls + other.llm_calls, self.prompt_tokens + other.prompt_tokens,
            self.output_tokens + other.output_tokens,
            self.thinking_tokens + other.thinking_tokens,
        )


@dataclass
class LLMResult:
    text: str
    model: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


@dataclass
class Answer:
    """What the pipeline returns. Everything needed to debug one request."""
    question: str
    text: str
    sources: list[Candidate] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    groundedness: float | None = None
    cache_hit: bool = False
    request_id: str = "-"
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    rewritten_question: str | None = None
    degraded: bool = False          # answered with part of the system unavailable
    degraded_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "text": self.text, "refused": self.refused,
            "refusal_reason": self.refusal_reason, "citations": self.citations,
            "groundedness": self.groundedness, "cache_hit": self.cache_hit,
            "request_id": self.request_id, "timings_ms": self.timings_ms,
            "rewritten_question": self.rewritten_question,
            "degraded": self.degraded, "degraded_reason": self.degraded_reason,
            "usage": asdict(self.usage), "sources": [c.explain(text_chars=600) for c in self.sources],
        }
