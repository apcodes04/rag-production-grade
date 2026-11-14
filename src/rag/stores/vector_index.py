"""Dense vector index with two interchangeable backends.

numpy  : exact brute force cosine. No dependency, no tuning, always correct.
         A matrix multiply over 100k vectors of 768 dims is a few milliseconds.
faiss  : the same exact search (IndexFlatIP) but SIMD optimised, and the door
         to approximate indexes (IVF, HNSW, PQ) when the corpus gets large.

Both expect L2 normalised vectors, so an inner product IS cosine similarity.
The honest production note: past a few million vectors you stop managing files
and move to a vector database (Qdrant, pgvector, Weaviate) for filtering,
replication and concurrent writes.
"""
from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, dim: int, backend: str = "numpy"):
        self.dim = dim
        self.backend = backend
        self.ids: list[str] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        self.ids.extend(ids)
        self._vectors = np.vstack([self._vectors, vectors]) if len(self._vectors) else vectors
        self._faiss = None                      # invalidate, rebuilt lazily on search

    def keep_only(self, keep_ids: set[str]) -> None:
        """Used by incremental ingestion when a document is deleted or replaced."""
        mask = [i for i, cid in enumerate(self.ids) if cid in keep_ids]
        self.ids = [self.ids[i] for i in mask]
        self._vectors = self._vectors[mask] if mask else np.zeros((0, self.dim), dtype=np.float32)
        self._faiss = None

    def _ensure_faiss(self):
        if self._faiss is None:
            import faiss  # imported only when actually used
            index = faiss.IndexFlatIP(self.dim)
            if len(self._vectors):
                index.add(self._vectors)
            self._faiss = index
        return self._faiss

    def search(self, query: np.ndarray, top_k: int,
               allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Top k by cosine similarity, optionally restricted to some chunks.

        The restriction is applied BEFORE the top k is taken, not after. Taking
        the global top k and filtering it would quietly return fewer results
        than asked for, and would miss a selected document entirely whenever
        other documents happen to score higher.
        """
        if not len(self.ids):
            return []
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores = (self._vectors @ query[0])     # cosine, vectors are normalised

        if allowed is not None:
            keep = np.array([cid in allowed for cid in self.ids], dtype=bool)
            if not keep.any():
                return []
            # -inf never survives a top k, so excluded chunks cannot be picked.
            scores = np.where(keep, scores, -np.inf)
            k = min(top_k, int(keep.sum()))
        else:
            k = min(top_k, len(self.ids))
            if self.backend == "faiss":
                faiss_scores, indices = self._ensure_faiss().search(query, k)
                return [(self.ids[i], float(sc))
                        for sc, i in zip(faiss_scores[0], indices[0], strict=True) if i != -1]

        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    def __len__(self) -> int:
        return len(self.ids)
