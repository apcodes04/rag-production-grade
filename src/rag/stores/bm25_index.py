"""BM25 Okapi, implemented here rather than imported, because the ranking
function is interview material and the tokenizer must be ours.

score(q, d) = sum over query terms of  IDF(t) * tf(t,d) * (k1 + 1)
                                       -----------------------------------------
                                       tf(t,d) + k1 * (1 - b + b * |d| / avgdl)

IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

Why the shape of the formula:
  tf saturates: the 10th occurrence of a word adds far less than the 2nd (k1
    controls how fast it saturates, typical 1.2 to 2.0).
  long documents are penalised: |d| / avgdl, softened by b (0 = no length
    normalisation, 1 = full, typical 0.75).
  rare terms count more: that is IDF. The "1 +" inside the log is Lucene's fix
    for the classic bug where a term appearing in more than half the documents
    gets a NEGATIVE idf and starts subtracting score.
"""
from __future__ import annotations

import math
from collections import defaultdict

from rag.tokenizer import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tokens: dict[str, list[str]] = {}          # chunk_id -> tokens
        self._postings: dict[str, dict[str, int]] = {}   # term -> {chunk_id: term frequency}
        self._length: dict[str, int] = {}                # chunk_id -> document length
        self._avgdl: float = 0.0

    # ---------------------------------------------------------------- build
    def add(self, chunk_id: str, text: str | None = None, tokens: list[str] | None = None) -> None:
        self._tokens[chunk_id] = tokens if tokens is not None else tokenize(text or "")

    def remove(self, chunk_id: str) -> None:
        self._tokens.pop(chunk_id, None)

    def build(self) -> None:
        """Build the inverted index. Called once after all documents are added."""
        postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._length = {}
        for chunk_id, tokens in self._tokens.items():
            self._length[chunk_id] = len(tokens)
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for term, freq in counts.items():
                postings[term][chunk_id] = freq
        self._postings = dict(postings)
        self._avgdl = (sum(self._length.values()) / len(self._length)) if self._length else 0.0

    # --------------------------------------------------------------- search
    def idf(self, term: str) -> float:
        n_docs = len(self._tokens)
        df = len(self._postings.get(term, ()))
        if n_docs == 0 or df == 0:
            return 0.0
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10,
               allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Score only the documents that contain a query term (that is what an
        inverted index is for: never touch the other N documents).

        `allowed` restricts the result to a subset of chunks, applied before
        the top k is taken so a selected document is never crowded out by one
        the caller excluded. IDF still comes from the whole corpus: term
        rarity is a property of the collection, not of the current filter.
        """
        if not self._postings:
            return []
        scores: dict[str, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for chunk_id, freq in postings.items():
                if allowed is not None and chunk_id not in allowed:
                    continue
                norm = 1.0 - self.b + self.b * (self._length[chunk_id] / (self._avgdl or 1.0))
                scores[chunk_id] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    # ------------------------------------------------------------ persistence
    def to_records(self) -> list[dict]:
        return [{"chunk_id": cid, "tokens": tokens} for cid, tokens in self._tokens.items()]

    @classmethod
    def from_records(cls, records: list[dict], k1: float = 1.5, b: float = 0.75) -> BM25Index:
        index = cls(k1=k1, b=b)
        for record in records:
            index.add(record["chunk_id"], tokens=record["tokens"])
        index.build()
        return index

    def __len__(self) -> int:
        return len(self._tokens)
