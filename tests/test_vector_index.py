import numpy as np
import pytest

from rag.providers.base import l2_normalise
from rag.stores.vector_index import VectorIndex


def make(backend="numpy"):
    index = VectorIndex(dim=3, backend=backend)
    vectors = l2_normalise(np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32))
    index.add(["a", "b", "c"], vectors)
    return index


def test_cosine_ranking_is_correct():
    hits = make().search(np.array([1, 0, 0], dtype=np.float32), top_k=3)
    assert [doc_id for doc_id, _ in hits] == ["a", "c", "b"]
    assert pytest.approx(hits[0][1], abs=1e-6) == 1.0
    assert pytest.approx(hits[1][1], abs=1e-4) == 0.7071


def test_top_k_is_capped_by_corpus_size():
    assert len(make().search(np.array([1, 0, 0], dtype=np.float32), top_k=99)) == 3


def test_empty_index_returns_nothing():
    assert VectorIndex(dim=3).search(np.zeros(3, dtype=np.float32), top_k=5) == []


def test_dimension_mismatch_is_rejected_loudly():
    with pytest.raises(ValueError):
        VectorIndex(dim=3).add(["a"], np.zeros((1, 5), dtype=np.float32))


def test_keep_only_drops_rows_and_ids_together():
    index = make()
    index.keep_only({"a", "c"})
    assert index.ids == ["a", "c"] and len(index.vectors) == 2
    assert [doc_id for doc_id, _ in index.search(np.array([0, 1, 0], dtype=np.float32), 2)] == ["c", "a"]


def test_faiss_backend_agrees_with_numpy():
    pytest.importorskip("faiss")
    query = l2_normalise(np.array([0.9, 0.3, 0.0], dtype=np.float32))
    numpy_hits = make("numpy").search(query, 3)
    faiss_hits = make("faiss").search(query, 3)
    assert [d for d, _ in numpy_hits] == [d for d, _ in faiss_hits]
    for (_, left), (_, right) in zip(numpy_hits, faiss_hits, strict=True):
        assert pytest.approx(left, abs=1e-5) == right
