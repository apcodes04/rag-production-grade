"""Shared fixtures. Every test runs fully offline on the fake providers, which
is why the suite is fast, free and deterministic."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.cache import AnswerCache
from rag.ingest import Ingestor
from rag.logging_setup import request_id_var
from rag.pipeline import RAGPipeline
from rag.providers.fake import FakeEmbeddings, FakeLLM
from rag.settings import Settings
from rag.stores.index import Index


@pytest.fixture(autouse=True)
def reset_request_id():
    """The request id lives in a ContextVar, which survives between tests in
    the same context. Resetting it keeps every test independent."""
    token = request_id_var.set("-")
    yield
    request_id_var.reset(token)


DOCS = {
    "refunds.txt": (
        "Refund policy.\n\nRefunds are issued within 30 days of purchase. "
        "To start a refund, contact the billing team with your order number. "
        "Refunds for digital goods are processed faster than physical returns.\n\n"
        "Partial refunds apply when only part of an order is returned."),
    "shipping.txt": (
        "Shipping information.\n\nStandard shipping takes five business days in metro areas "
        "and eight business days elsewhere. Express shipping delivers next day for an extra fee. "
        "Tracking numbers are emailed once a parcel leaves the warehouse."),
    "security.txt": (
        "Account security.\n\nTwo factor authentication is required for admin accounts. "
        "Passwords must be at least twelve characters long and are rotated every ninety days. "
        "Suspicious logins trigger an email alert to the account owner."),
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, text in DOCS.items():
        (raw / name).write_text(text, encoding="utf-8")
    return raw


@pytest.fixture
def settings(tmp_path: Path, corpus: Path) -> Settings:
    return Settings(
        _env_file=None, provider="fake", embedding_dim=128,
        raw_dir=corpus, index_dir=tmp_path / "index", cache_dir=tmp_path / "cache",
        # Every path a test can write to must live under tmp_path. Leaving
        # upload_dir at its default pointed the API tests at the real
        # data/uploads/ folder, so they wrote into the project and later runs
        # of unrelated tests then found those files.
        upload_dir=tmp_path / "uploads",
        chunk_size=80, chunk_overlap=16, top_k_dense=10, top_k_sparse=10,
        top_k_fused=10, top_k_final=3, rerank_input_k=10,
        min_dense_score=0.30, min_sparse_score=1.0, log_level="WARNING")


@pytest.fixture
def embedder(settings: Settings) -> FakeEmbeddings:
    return FakeEmbeddings(dim=settings.embedding_dim)


@pytest.fixture
def index(settings: Settings, embedder: FakeEmbeddings) -> Index:
    index = Index(embedder.dim, embedder.name)
    Ingestor(index, embedder, settings).ingest_dir()
    return index


@pytest.fixture
def make_pipeline(settings: Settings, index: Index, embedder: FakeEmbeddings):
    def build(llm: FakeLLM | None = None, **overrides) -> RAGPipeline:
        active = settings.model_copy(update=overrides) if overrides else settings
        cache = AnswerCache(active.cache_dir / "answers.sqlite3", active.cache_ttl_s)
        return RAGPipeline(active, index, embedder, llm or FakeLLM(), cache=cache)
    return build


@pytest.fixture
def pipeline(make_pipeline) -> RAGPipeline:
    return make_pipeline()
