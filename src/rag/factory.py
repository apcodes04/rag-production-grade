"""Composition root: the one place that decides which concrete classes to use.

Everything else in the codebase depends on interfaces, so this file is the
only one that imports Gemini. Swapping providers, or running the whole system
offline for tests, is a one line change here.
"""
from __future__ import annotations

from rag.cache import AnswerCache
from rag.pipeline import RAGPipeline
from rag.providers.fake import FakeEmbeddings, FakeLLM
from rag.providers.rate_limit import build_limiter
from rag.settings import Settings
from rag.stores.index import Index


def build_embedder(settings: Settings):
    if settings.provider == "fake":
        return FakeEmbeddings(dim=settings.embedding_dim)
    from rag.providers.gemini import GeminiEmbeddings
    return GeminiEmbeddings(
        settings.gemini_api_key.get_secret_value(), settings.embedding_model,
        settings.embedding_dim, batch_size=settings.embed_batch_size,
        max_retries=settings.embed_max_retries, timeout_s=settings.embed_timeout_s,
        deadline_s=settings.embed_deadline_s,
        limiter=build_limiter(settings.embed_requests_per_minute))


def build_llm(settings: Settings):
    if settings.provider == "fake":
        return FakeLLM()
    from rag.providers.gemini import GeminiLLM

    # One bucket per model: quotas are per model, and the fallback exists
    # precisely because the primary's bucket is empty.
    def gemini(model: str):
        return GeminiLLM(settings.gemini_api_key.get_secret_value(), model,
                         max_retries=settings.llm_max_retries, timeout_s=settings.llm_timeout_s,
                         thinking_level=settings.llm_thinking_level,
                         deadline_s=settings.llm_deadline_s,
                         limiter=build_limiter(settings.llm_requests_per_minute))

    primary = gemini(settings.llm_model)
    if not settings.llm_fallback_model:
        return primary
    from rag.providers.fallback import FallbackLLM
    return FallbackLLM(primary, gemini(settings.llm_fallback_model))


def load_index(settings: Settings, embedder) -> Index:
    return Index.load(settings.index_dir, dim=embedder.dim,
                      embedding_model=embedder.name, backend=settings.vector_backend)


def build_pipeline(settings: Settings, *, index: Index | None = None) -> RAGPipeline:
    embedder = build_embedder(settings)
    llm = build_llm(settings)
    index = index if index is not None else load_index(settings, embedder)
    cache = AnswerCache(settings.cache_dir / "answers.sqlite3",
                        settings.cache_ttl_s) if settings.cache_enabled else None
    return RAGPipeline(settings, index, embedder, llm, cache=cache)
