import json

import numpy as np
import pytest

from rag.errors import StoreError
from rag.stores.index import Index
from rag.types import Chunk


def chunk(n: int) -> Chunk:
    return Chunk(chunk_id=f"doc-{n:05d}", doc_id="doc", source="a.txt", page=1,
                 chunk_index=n, text=f"chunk number {n} about refunds", token_count=6)


def filled(dim: int = 4) -> Index:
    index = Index(dim, "fake-embeddings")
    chunks = [chunk(i) for i in range(3)]
    vectors = np.eye(3, dim, dtype=np.float32)
    index.add(chunks, vectors)
    return index


def test_save_creates_a_new_version_each_time(tmp_path):
    index = filled()
    first = index.save(tmp_path)
    second = index.save(tmp_path)
    assert first.name == "v1" and second.name == "v2"
    assert json.loads((tmp_path / "current.json").read_text())["version"] == "v2"
    # the old version is still on disk, so a rollback is one line in current.json
    assert (tmp_path / "v1" / "manifest.json").exists()


def test_round_trip_preserves_chunks_vectors_and_bm25(tmp_path):
    index = filled()
    index.save(tmp_path)
    loaded = Index.load(tmp_path, dim=4, embedding_model="fake-embeddings")
    assert loaded.chunks.keys() == index.chunks.keys()
    assert np.allclose(loaded.vectors.vectors, index.vectors.vectors)
    assert loaded.bm25.search("refunds") and loaded.index_version() == index.index_version()


def test_loading_an_empty_directory_is_not_an_error(tmp_path):
    assert len(Index.load(tmp_path, dim=4, embedding_model="fake-embeddings").chunks) == 0


def test_model_or_dim_mismatch_refuses_to_load(tmp_path):
    filled().save(tmp_path)
    with pytest.raises(StoreError, match="dim"):
        Index.load(tmp_path, dim=8, embedding_model="fake-embeddings")
    with pytest.raises(StoreError):
        Index.load(tmp_path, dim=4, embedding_model="some-other-model")


def test_index_version_changes_with_content(tmp_path):
    index = filled()
    before = index.index_version()
    index.add([chunk(99)], np.eye(1, 4, dtype=np.float32))
    assert index.index_version() != before


def test_remove_doc_cleans_all_three_structures(tmp_path):
    index = filled()
    index.docs["a.txt"] = __import__("rag.stores.index", fromlist=["DocRecord"]).DocRecord(
        path="a.txt", doc_id="doc", sha256="x", size=1, mtime_ns=1,
        chunk_ids=[c for c in index.chunks])
    removed = index.remove_doc("a.txt")
    assert removed == 3
    assert not index.chunks and len(index.vectors) == 0 and len(index.bm25) == 0


def test_no_pickle_files_are_written(tmp_path):
    filled().save(tmp_path)
    assert not list(tmp_path.rglob("*.pkl"))
