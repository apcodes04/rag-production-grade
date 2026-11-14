"""Checks that run AFTER the model answers, before the user sees it.

An LLM given five chunks will usually write a confident answer even when the
chunks do not support it. Two cheap, deterministic checks catch most of it:

  citation validity: every [n] the model wrote must point at a source we
    actually sent. A model citing [7] when it got 5 sources is hallucinating
    its evidence, which is worse than having no citation at all.
  groundedness: what share of the answer's content words appear in the cited
    chunks. Crude but surprisingly effective, costs nothing, and never calls
    a model. The production upgrade is an NLI model or an LLM judge, which is
    what the evaluation harness uses offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.tokenizer import tokenize

CITATION = re.compile(r"\[(\d{1,2})\]")
REFUSAL_HINTS = ("i don't know", "i do not know", "not in the provided context",
                 "cannot find", "does not contain", "no information")


@dataclass
class OutputVerdict:
    ok: bool
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    groundedness: float = 0.0
    reason: str | None = None


def parse_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    seen, invalid = [], []
    for match in CITATION.findall(answer):
        number = int(match)
        target = seen if 1 <= number <= n_sources else invalid
        if number not in target:
            target.append(number)
    return sorted(seen), sorted(invalid)


def groundedness(answer: str, source_texts: list[str]) -> float:
    answer_terms = set(tokenize(answer))
    if not answer_terms:
        return 0.0
    source_terms = set()
    for text in source_texts:
        source_terms.update(tokenize(text))
    return len(answer_terms & source_terms) / len(answer_terms)


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(hint in lowered for hint in REFUSAL_HINTS)


def check_answer(answer: str, source_texts: list[str], *, require_citations: bool,
                 min_groundedness: float) -> OutputVerdict:
    if not answer.strip():
        return OutputVerdict(False, reason="empty_answer")
    citations, invalid = parse_citations(answer, len(source_texts))
    # A model that honestly says "the context does not cover this" is doing the
    # right thing: do not punish it for having no citations.
    if looks_like_refusal(answer):
        return OutputVerdict(True, citations, invalid, 1.0, reason="model_refused")
    if invalid:
        return OutputVerdict(False, citations, invalid, 0.0, reason="invalid_citation")
    if require_citations and not citations:
        return OutputVerdict(False, citations, invalid, 0.0, reason="missing_citations")
    cited_texts = [source_texts[i - 1] for i in citations] or source_texts
    score = groundedness(answer, cited_texts)
    if score < min_groundedness:
        return OutputVerdict(False, citations, invalid, score, reason="ungrounded")
    return OutputVerdict(True, citations, invalid, score)
