"""Hybrid retrieval: dense (meaning) plus sparse (exact words), then fusion.

Why both, in one sentence each:
  dense  finds "how do I get my money back" in a chunk that says "refund
         policy", because the vectors are close in meaning.
  sparse finds "error E4012" or "Section 7.2" or a person's surname, where
         the exact token is the whole point and embeddings are blurry.
Nobody ships only one. Hybrid is the default in every serious RAG system.
"""
from __future__ import annotations

import logging

from rag.errors import RetryableProviderError
from rag.providers.base import EmbeddingProvider
from rag.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from rag.settings import Settings
from rag.stores.index import Index
from rag.types import Candidate, RetrievalResult

log = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, index: Index, embedder: EmbeddingProvider, settings: Settings):
        self.index = index
        self.embedder = embedder
        self.settings = settings

    def allowed_chunks(self, sources: list[str] | None) -> set[str] | None:
        """Chunk ids belonging to the named documents, or None for no filter.

        An unknown name contributes nothing rather than raising: a document
        can be deleted between the moment a page lists it and the moment a
        question is asked, and that should narrow the search, not fail it.
        """
        if not sources:
            return None
        wanted = set(sources)
        return {chunk_id
                for path, record in self.index.docs.items() if path in wanted
                for chunk_id in record.chunk_ids}

    def retrieve(self, query: str, sources: list[str] | None = None) -> RetrievalResult:
        cfg = self.settings
        degraded_reason = None
        allowed = self.allowed_chunks(sources)
        try:
            dense_hits = self.index.vectors.search(self.embedder.embed_query(query),
                                                   cfg.top_k_dense, allowed=allowed)
        except RetryableProviderError as exc:
            # The embedding API is down. BM25 needs no network, so keyword only
            # retrieval still answers many questions. Degrade loudly: a quietly
            # worse system is harder to debug than one that is simply down.
            if not cfg.allow_degraded_retrieval:
                raise
            log.warning("embedding provider unavailable, keyword-only retrieval",
                        extra={"error": str(exc)[:200]})
            dense_hits, degraded_reason = [], "embeddings_unavailable"
        sparse_hits = self.index.bm25.search(query, cfg.top_k_sparse, allowed=allowed)

        candidates: dict[str, Candidate] = {}

        def slot(chunk_id: str) -> Candidate | None:
            chunk = self.index.chunks.get(chunk_id)
            if chunk is None:                   # index out of sync, skip rather than crash
                log.warning("hit for unknown chunk", extra={"chunk_id": chunk_id})
                return None
            return candidates.setdefault(chunk_id, Candidate(chunk=chunk))

        for rank, (chunk_id, score) in enumerate(dense_hits, start=1):
            candidate = slot(chunk_id)
            if candidate:
                candidate.dense_score, candidate.rank_dense = score, rank
        for rank, (chunk_id, score) in enumerate(sparse_hits, start=1):
            candidate = slot(chunk_id)
            if candidate:
                candidate.sparse_score, candidate.rank_sparse = score, rank

        if cfg.fusion == "rrf":
            fused = reciprocal_rank_fusion(
                [[cid for cid, _ in dense_hits], [cid for cid, _ in sparse_hits]],
                k=cfg.rrf_k, weights=[cfg.dense_weight, 1.0 - cfg.dense_weight])
        else:
            fused = weighted_fusion([dict(dense_hits), dict(sparse_hits)],
                                    [cfg.dense_weight, 1.0 - cfg.dense_weight])

        for chunk_id, score in fused.items():
            if chunk_id in candidates:
                candidates[chunk_id].fused_score = score

        ranked = sorted(candidates.values(), key=lambda c: (-(c.fused_score or 0.0),
                                                            c.chunk.chunk_id))
        ranked = ranked[: cfg.top_k_fused]
        for position, candidate in enumerate(ranked, start=1):
            candidate.final_rank = position
        log.info("retrieved", extra={"dense": len(dense_hits), "sparse": len(sparse_hits),
                                     "fused": len(ranked), "fusion": cfg.fusion,
                                     "degraded": bool(degraded_reason)})
        return RetrievalResult(ranked, bool(degraded_reason), degraded_reason)

    def is_relevant_enough(self, candidates: list[Candidate]) -> bool:
        """The refusal gate. Without it the LLM always gets five chunks, even
        for a question the corpus cannot answer, and it will invent something."""
        cfg = self.settings
        if len(candidates) < cfg.min_candidates:
            return False
        best_dense = max((c.dense_score or -1.0) for c in candidates) if candidates else -1.0
        best_sparse = max((c.sparse_score or 0.0) for c in candidates) if candidates else 0.0
        return best_dense >= cfg.min_dense_score or best_sparse >= cfg.min_sparse_score
