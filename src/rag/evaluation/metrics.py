"""Retrieval metrics. All of them take one list: was the document at rank i relevant?

recall@k     of all the relevant chunks that exist, how many did we put in the
             top k. This is THE retrieval metric for RAG: the generator can
             only use what retrieval found.
precision@k  of the k we returned, how many were relevant. Matters because
             every irrelevant chunk is noise in the prompt and money burnt.
MRR          1 / rank of the first relevant hit. Rewards putting the right
             chunk first, which matters when top_k_final is small.
nDCG@k       like MRR but credits every relevant hit, discounted by position.
hit rate     did we find anything at all. The crudest sanity check.
"""
from __future__ import annotations

import math


def hit_rate(relevance: list[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def precision_at_k(relevance: list[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(relevance[:k]) / k


def recall_at_k(relevance: list[bool], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return min(sum(relevance[:k]) / total_relevant, 1.0)


def mrr(relevance: list[bool]) -> float:
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: list[bool], k: int, total_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, is_relevant in enumerate(relevance[:k], start=1) if is_relevant)
    ideal_hits = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return round(ordered[int(position)], 4)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 4)
