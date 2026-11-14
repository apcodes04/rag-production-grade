import math

from rag.stores.bm25_index import BM25Index


def build(docs: dict[str, str], **kwargs) -> BM25Index:
    index = BM25Index(**kwargs)
    for doc_id, text in docs.items():
        index.add(doc_id, text)
    index.build()
    return index


CORPUS = {
    "a": "refund policy refunds are issued within 30 days",
    "b": "shipping takes five business days",
    "c": "support answers on weekdays",
}


def test_matches_on_exact_term_and_ignores_others():
    results = dict(build(CORPUS).search("refund"))
    assert "a" in results and results["a"] > 0
    assert "c" not in results


def test_idf_is_never_negative_even_for_common_terms():
    # The classic BM25 bug: with the textbook IDF a term in more than half the
    # documents gets a negative weight and starts subtracting score.
    index = build({"a": "cat sat", "b": "cat ran", "c": "cat flew"})
    assert index.idf("cat") >= 0
    assert all(score >= 0 for _, score in index.search("cat"))


def test_rare_terms_outscore_common_ones():
    index = build({"a": "common term", "b": "common term", "c": "common rare"})
    assert index.idf("rare") > index.idf("common")


def test_term_frequency_saturates():
    index = build({"once": "alpha beta", "many": "alpha alpha alpha alpha alpha alpha beta"})
    scores = dict(index.search("alpha"))
    # ten times the occurrences must NOT give ten times the score
    assert scores["many"] > scores["once"]
    assert scores["many"] < 3 * scores["once"]


def test_length_normalisation_prefers_the_shorter_document():
    padding = " ".join(f"filler{i}" for i in range(80))
    index = build({"short": "quantum tunnelling explained",
                   "long": f"quantum tunnelling explained {padding}"})
    scores = dict(index.search("quantum tunnelling"))
    assert scores["short"] > scores["long"]
    # b = 0 switches length normalisation off, so the gap must shrink
    flat = dict(build({"short": "quantum tunnelling explained",
                       "long": f"quantum tunnelling explained {padding}"}, b=0.0).search("quantum tunnelling"))
    assert math.isclose(flat["short"], flat["long"], rel_tol=1e-9)


def test_unknown_terms_and_empty_index_are_safe():
    assert build(CORPUS).search("helicopter") == []
    assert BM25Index().search("anything") == []


def test_remove_and_rebuild():
    index = build(CORPUS)
    index.remove("a")
    index.build()
    assert "a" not in dict(index.search("refund"))
    assert len(index) == 2


def test_round_trip_through_records():
    index = build(CORPUS)
    restored = BM25Index.from_records(index.to_records())
    assert index.search("refund") == restored.search("refund")
