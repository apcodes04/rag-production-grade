import pytest

from rag.errors import RetryableProviderError
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker, IdentityReranker, build_reranker
from rag.types import Candidate, Chunk


def retriever(settings, index, embedder) -> HybridRetriever:
    return HybridRetriever(index, embedder, settings)


def test_dense_and_sparse_both_contribute(settings, index, embedder):
    results = retriever(settings, index, embedder).retrieve("refund within 30 days").candidates
    assert results
    assert any(c.dense_score is not None for c in results)
    assert any(c.sparse_score is not None for c in results)
    assert results[0].chunk.source == "refunds.txt"


def test_results_are_ordered_by_fused_score(settings, index, embedder):
    results = retriever(settings, index, embedder).retrieve("express shipping").candidates
    scores = [c.fused_score for c in results]
    assert scores == sorted(scores, reverse=True)
    assert [c.final_rank for c in results] == list(range(1, len(results) + 1))


def test_exact_rare_term_is_found_even_when_meaning_is_blurry(settings, index, embedder):
    # This is why sparse retrieval stays in the system.
    results = retriever(settings, index, embedder).retrieve("twelve characters").candidates
    assert results[0].chunk.source == "security.txt"


def test_relevance_gate_rejects_off_topic_questions(settings, index, embedder):
    searcher = retriever(settings, index, embedder)
    assert searcher.is_relevant_enough(searcher.retrieve("refund policy").candidates)
    assert not searcher.is_relevant_enough(searcher.retrieve("who won the football world cup").candidates)


def test_weighted_fusion_is_an_option(settings, index, embedder):
    weighted = settings.model_copy(update={"fusion": "weighted"})
    results = HybridRetriever(index, embedder, weighted).retrieve("refund").candidates
    assert results and results[0].fused_score is not None


def _candidates(texts):
    return [Candidate(chunk=Chunk(f"c{i}", "d", "s.txt", 1, i, text, 5))
            for i, text in enumerate(texts)]


def test_identity_reranker_keeps_order_and_truncates():
    candidates = _candidates(["one", "two", "three"])
    top = IdentityReranker().rerank("q", candidates, top_k=2)
    assert [c.chunk.text for c in top] == ["one", "two"]
    assert [c.final_rank for c in top] == [1, 2]


def test_cross_encoder_reorders_and_sets_scores(monkeypatch):
    reranker = CrossEncoderReranker("dummy-model")

    class Stub:
        def predict(self, pairs):
            return [0.1, 0.9, 0.5]           # second pair is the best match
    monkeypatch.setattr(reranker, "_load", lambda: Stub())
    top = reranker.rerank("q", _candidates(["a", "b", "c"]), top_k=2)
    assert [c.chunk.text for c in top] == ["b", "c"]
    assert top[0].rerank_score == pytest.approx(0.9)


def test_a_broken_reranker_never_breaks_a_query(monkeypatch):
    reranker = CrossEncoderReranker("dummy-model")

    def explode():
        raise RuntimeError("model file missing")
    monkeypatch.setattr(reranker, "_load", explode)
    top = reranker.rerank("q", _candidates(["a", "b"]), top_k=2)
    assert [c.chunk.text for c in top] == ["a", "b"]      # fell back to fusion order


def test_build_reranker_follows_settings(settings):
    assert isinstance(build_reranker(settings), IdentityReranker)
    assert isinstance(build_reranker(settings.model_copy(update={"reranker": "cross_encoder"})),
                      CrossEncoderReranker)


def test_a_dead_embedding_api_degrades_to_keyword_search(settings, index, embedder):
    """BM25 needs no network, so an embedding outage is a quality problem, not
    an outage. It must be visible, never silent."""
    class DeadEmbedder:
        name, dim, usage = embedder.name, embedder.dim, embedder.usage

        def embed_query(self, text):
            raise RetryableProviderError("503 UNAVAILABLE")

        def embed_documents(self, texts):
            raise RetryableProviderError("503 UNAVAILABLE")

    result = HybridRetriever(index, DeadEmbedder(), settings).retrieve("refund policy")
    assert result.degraded and result.degraded_reason == "embeddings_unavailable"
    assert result.candidates and result.candidates[0].chunk.source == "refunds.txt"
    assert all(c.dense_score is None for c in result.candidates)


def test_degrading_can_be_switched_off(settings, index, embedder):
    class DeadEmbedder:
        name, dim, usage = embedder.name, embedder.dim, embedder.usage

        def embed_query(self, text):
            raise RetryableProviderError("503 UNAVAILABLE")

    strict = settings.model_copy(update={"allow_degraded_retrieval": False})
    with pytest.raises(RetryableProviderError):
        HybridRetriever(index, DeadEmbedder(), strict).retrieve("refund policy")
