"""Restricting retrieval to chosen documents.

The filter is applied inside each search, before the top k is taken. Taking
the global top k and filtering afterwards would silently return fewer results
than asked for, and would miss a chosen document entirely whenever other
documents happen to score higher. These tests pin that behaviour.
"""
from __future__ import annotations

from rag.cache import AnswerCache


def sources_of(answer):
    return {c.chunk.source for c in answer.sources}


def test_an_answer_can_be_restricted_to_one_document(pipeline):
    answer = pipeline.answer("How long do refunds take?", sources=["refunds.txt"])
    assert sources_of(answer) == {"refunds.txt"}


def test_the_filter_excludes_the_document_that_would_otherwise_win(pipeline):
    """The honest test: ask something refunds.txt answers, but allow only
    shipping.txt. Every chunk must come from shipping.txt."""
    unfiltered = pipeline.answer("How long do refunds take?")
    assert "refunds.txt" in sources_of(unfiltered)

    filtered = pipeline.answer("How long do refunds take?", sources=["shipping.txt"])
    assert "refunds.txt" not in sources_of(filtered)
    assert sources_of(filtered) <= {"shipping.txt"}


def test_filtering_still_returns_a_full_slate_of_candidates(pipeline, index):
    """Filter first, then take top k. If it were the other way round, asking
    for one small document would return almost nothing."""
    result = pipeline.retriever.retrieve("shipping", sources=["shipping.txt"])
    shipping_chunks = sum(1 for c in index.chunks.values() if c.source == "shipping.txt")
    assert len(result.candidates) == min(shipping_chunks, pipeline.settings.top_k_fused)
    assert {c.chunk.source for c in result.candidates} == {"shipping.txt"}


def test_an_unknown_document_name_narrows_rather_than_crashes(pipeline):
    """A document can be deleted between the page listing it and the question
    being asked. That should narrow the search, not raise."""
    answer = pipeline.answer("How long do refunds take?", sources=["deleted-since.txt"])
    assert answer.sources == []
    assert answer.refused is True


def test_no_filter_still_searches_everything(pipeline, index):
    result = pipeline.retriever.retrieve("refund shipping security")
    assert len({c.chunk.source for c in result.candidates}) > 1


def test_the_document_filter_is_part_of_the_cache_key():
    """Same question, different documents, is a different question. If the key
    ignored the filter, the first answer would be replayed for the second."""
    common = dict(history=[], index_version="v1", prompt_version="p1", config={})
    whole = AnswerCache.make_key("what is the policy?", **common)
    one = AnswerCache.make_key("what is the policy?", **common, sources=["refunds.txt"])
    other = AnswerCache.make_key("what is the policy?", **common, sources=["shipping.txt"])

    assert len({whole, one, other}) == 3
    # Order of the same set must not change the key.
    assert AnswerCache.make_key("q", **common, sources=["a.txt", "b.txt"]) == \
           AnswerCache.make_key("q", **common, sources=["b.txt", "a.txt"])


def test_a_filtered_answer_is_not_served_to_an_unfiltered_question(pipeline):
    first = pipeline.answer("How long do refunds take?", sources=["shipping.txt"])
    second = pipeline.answer("How long do refunds take?")
    assert second.cache_hit is False
    assert "refunds.txt" in sources_of(second)
    assert first.text != second.text or not first.refused
