"""Merging a dense ranking and a sparse ranking into one.

The problem: cosine similarity lives in [-1, 1] and BM25 is unbounded (0 to
30+, corpus dependent). Adding them directly means BM25 silently dominates.

Two honest ways out:
  RRF      throw the scores away, keep only the ranks. score = sum 1/(k+rank).
           No normalisation, no tuning, robust across corpora. k (60 by
           convention) flattens the curve so rank 1 does not crush rank 2.
  weighted min-max normalise each score list to [0, 1] inside the result set,
           then take a weighted sum. Keeps score magnitude (a document that
           dense retrieval loves by a mile stays ahead), but the normalisation
           is relative to this result set, so it is less stable.
Default is RRF. Switch with settings.fusion and measure on the golden set.
"""
from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: list[list[str]], *, k: int = 60,
                           weights: list[float] | None = None) -> dict[str, float]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ranking, weight in zip(ranked_lists, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


def min_max_normalise(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def weighted_fusion(score_lists: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for scores, weight in zip(score_lists, weights, strict=True):
        for doc_id, value in min_max_normalise(scores).items():
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * value
    return fused
