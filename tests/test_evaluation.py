import json

from rag.evaluation.harness import GoldenQuestion, evaluate, format_report, load_golden
from rag.evaluation.metrics import mrr, ndcg_at_k, percentile, precision_at_k, recall_at_k

GOLDEN = [
    GoldenQuestion(id="g1", question="How long do refunds take?",
                   evidence_phrase="within 30 days", must_include=["30"]),
    GoldenQuestion(id="g2", question="How long does express shipping take?",
                   evidence_phrase="Express shipping delivers next day",
                   must_include=["next day"]),
    GoldenQuestion(id="g3", question="Who won the football world cup in 1998?",
                   evidence_phrase="", expect_refusal=True),
]


def test_metric_formulas():
    assert recall_at_k([True, False, True], k=3, total_relevant=2) == 1.0
    assert recall_at_k([False, True], k=1, total_relevant=2) == 0.0
    assert precision_at_k([True, False, False, False], k=4) == 0.25
    assert mrr([False, False, True]) == 1 / 3
    assert mrr([False, False]) == 0.0
    assert ndcg_at_k([True, False], k=2, total_relevant=1) == 1.0
    assert ndcg_at_k([False, True], k=2, total_relevant=1) < 1.0
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_evaluate_reports_retrieval_answer_and_refusal_quality(pipeline):
    report = evaluate(pipeline, GOLDEN)
    aggregates = report["aggregates"]
    assert aggregates["questions"] == 3 and aggregates["answerable"] == 2
    assert aggregates["recall@5"] == 1.0 and aggregates["mrr"] == 1.0
    assert aggregates["correct_refusals"] == 1.0 and aggregates["false_refusals"] == 0.0
    assert aggregates["citation_rate"] == 1.0 and aggregates["groundedness"] > 0.5
    assert aggregates["latency_p95_ms"] >= aggregates["latency_p50_ms"]


def test_report_carries_the_config_so_two_runs_are_comparable(pipeline):
    report = evaluate(pipeline, GOLDEN)
    assert report["config"]["chunk_size"] == pipeline.settings.chunk_size
    assert report["index"]["chunks"] == len(pipeline.index.chunks)
    assert "suggested_min_dense_score" in report["calibration"]
    assert "AGGREGATES" in format_report(report)
    json.dumps(report)                      # the report must be serialisable


def test_the_shipped_golden_set_is_valid(tmp_path):
    questions = load_golden("eval/golden_set.json")
    assert len(questions) >= 10
    assert all(q.evidence_phrase or q.expect_refusal for q in questions)
    assert len({q.id for q in questions}) == len(questions)


def test_one_failing_question_does_not_kill_the_run(pipeline, monkeypatch):
    from rag.errors import RetryableProviderError
    real = pipeline.answer
    calls = {"n": 0}

    def flaky(question, history=None, *, use_cache=True):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RetryableProviderError("503 UNAVAILABLE")
        return real(question, history, use_cache=use_cache)

    monkeypatch.setattr(pipeline, "answer", flaky)
    report = evaluate(pipeline, GOLDEN)
    assert report["aggregates"]["errors"] == 1
    assert report["aggregates"]["questions"] == 3
    failed = [r for r in report["results"] if r["error"]]
    assert len(failed) == 1 and "503" in failed[0]["error"]
    assert "ERROR" in format_report(report)


def test_running_in_parallel_gives_the_same_results(pipeline):
    serial = evaluate(pipeline, GOLDEN, workers=1)
    parallel = evaluate(pipeline, GOLDEN, workers=4)
    assert [r["id"] for r in serial["results"]] == [r["id"] for r in parallel["results"]]
    assert serial["aggregates"]["recall@5"] == parallel["aggregates"]["recall@5"]
