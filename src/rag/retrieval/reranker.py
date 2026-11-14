"""Reranking: a second, slower, more accurate pass over the top candidates.

bi-encoder (what retrieval uses): query and chunk are embedded SEPARATELY and
  compared by cosine. Fast, because every chunk vector is precomputed, but the
  model never sees the two texts together.
cross-encoder (a real reranker): the pair (query, chunk) goes through the
  model TOGETHER and it outputs one relevance score. Far more accurate on
  subtle matches, and far slower, so it only ever runs on the top 20 to 50.

The reference project called its reranker a cross-encoder in the config and
then re-scored with the SAME embedding model used for retrieval. That adds an
API call per query and almost no information, because it is the same geometry.
Default here is "none" so the system runs without torch; switch to
"cross_encoder" (pip install sentence-transformers) and measure the change on
the golden set before believing it helps.
"""
from __future__ import annotations

import logging
from typing import Protocol

from rag.settings import Settings
from rag.types import Candidate

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        ...


class IdentityReranker:
    """Keeps the fusion order. The honest default when no reranker is configured."""
    name = "none"

    def rerank(self, query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        selected = candidates[:top_k]
        for position, candidate in enumerate(selected, start=1):
            candidate.final_rank = position
        return selected


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # heavy, so imported lazily
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        if not candidates:
            return []
        try:
            scores = self._load().predict([(query, c.chunk.text) for c in candidates])
        except Exception as exc:                 # noqa: BLE001 never fail a query on the reranker
            log.warning("reranker failed, falling back to fusion order",
                        extra={"error": str(exc)[:200]})
            return IdentityReranker().rerank(query, candidates, top_k)
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)
        ranked = sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))[:top_k]
        for position, candidate in enumerate(ranked, start=1):
            candidate.final_rank = position
        return ranked


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "cross_encoder":
        return CrossEncoderReranker(settings.reranker_model)
    return IdentityReranker()
