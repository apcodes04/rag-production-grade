from rag.retrieval.fusion import min_max_normalise, reciprocal_rank_fusion, weighted_fusion


def test_rrf_matches_the_formula():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert round(scores["a"], 10) == round(1 / 61 + 1 / 62, 10)
    assert round(scores["a"], 10) == round(scores["b"], 10)


def test_rrf_rewards_appearing_in_both_lists():
    scores = reciprocal_rank_fusion([["a", "x"], ["y", "a"]], k=60)
    assert scores["a"] > scores["x"] and scores["a"] > scores["y"]


def test_rrf_ignores_score_scale():
    # This is the whole point of RRF: BM25 scores of 30 and cosine scores of
    # 0.4 cannot be added directly, but their RANKS can.
    dense = ["d1", "d2"]
    sparse = ["s1", "s2"]
    first = reciprocal_rank_fusion([dense, sparse], k=60)
    second = reciprocal_rank_fusion([dense, sparse], k=60)
    assert first == second and first["d1"] == first["s1"]


def test_smaller_k_sharpens_the_top_rank():
    flat = reciprocal_rank_fusion([["a", "b"]], k=60)
    sharp = reciprocal_rank_fusion([["a", "b"]], k=1)
    assert sharp["a"] / sharp["b"] > flat["a"] / flat["b"]


def test_min_max_normalise_handles_a_flat_list():
    assert min_max_normalise({"a": 5.0, "b": 5.0}) == {"a": 1.0, "b": 1.0}
    assert min_max_normalise({}) == {}


def test_weighted_fusion_respects_weights():
    dense = {"a": 1.0, "b": 0.0}
    sparse = {"a": 0.0, "b": 1.0}
    assert weighted_fusion([dense, sparse], [0.9, 0.1])["a"] > weighted_fusion([dense, sparse], [0.9, 0.1])["b"]
    assert weighted_fusion([dense, sparse], [0.1, 0.9])["b"] > weighted_fusion([dense, sparse], [0.1, 0.9])["a"]
