"""The evaluation harness: the only honest way to change anything in RAG.

Without this file, "is chunk size 512 better than 256?" is an opinion. With
it, it is a number you can put in a pull request.

Ground truth without brittle labels: each golden question carries an
`evidence_phrase`, a short exact string that appears only in the chunks that
truly answer it. A chunk counts as relevant when it contains that phrase. The
labels therefore survive re-chunking, re-embedding and re-indexing, which
hand written chunk ids do not.

Reported per run: retrieval quality (recall, precision, MRR, nDCG), answer
quality (citation validity, groundedness, keyword coverage), refusal
behaviour on deliberately out of scope questions, latency percentiles, cost,
and the score distributions you need to calibrate the refusal threshold.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.errors import RagError
from rag.evaluation.metrics import (
    hit_rate,
    mean,
    mrr,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
)
from rag.pipeline import RAGPipeline

K_VALUES = (1, 3, 5, 10)

JUDGE_SYSTEM = (
    "You grade whether an ANSWER is fully supported by the SOURCES.\n"
    "Reply with one number between 0 and 1 and nothing else.\n"
    "1 = every claim is supported, 0 = the answer invents facts."
)


@dataclass
class GoldenQuestion:
    id: str
    question: str
    evidence_phrase: str = ""
    must_include: list[str] = field(default_factory=list)
    expect_refusal: bool = False


@dataclass
class QuestionResult:
    id: str
    question: str
    expect_refusal: bool
    refused: bool
    refusal_reason: str | None
    retrieval: dict[str, float]
    best_dense: float
    best_sparse: float
    n_relevant_in_corpus: int
    answer: str
    citations: list[int]
    groundedness: float | None
    keyword_coverage: float
    judge_score: float | None
    latency_ms: float
    error: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)


def load_golden(path: Path) -> list[GoldenQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenQuestion(**item) for item in data["questions"]]


def _keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lowered = answer.lower()
    return sum(1 for word in keywords if word.lower() in lowered) / len(keywords)


def judge_faithfulness(llm, question: str, answer: str, sources: list[str]) -> float | None:
    """LLM as a judge. Slower and costs money, so it is opt in. Never used as
    the only signal: a judge from the same model family shares its blind spots."""
    try:
        joined = "\n\n".join(sources)
        result = llm.generate(JUDGE_SYSTEM,
                              f"QUESTION:\n{question}\n\nSOURCES:\n{joined}\n\nANSWER:\n{answer}\n\nScore:",
                              max_output_tokens=8, temperature=0.0)
        return max(0.0, min(1.0, float(result.text.strip().split()[0])))
    except Exception:                       # noqa: BLE001 a broken judge must not break the run
        return None


def _empty_retrieval() -> dict[str, float]:
    metrics = {}
    for k in K_VALUES:
        metrics |= {f"recall@{k}": 0.0, f"precision@{k}": 0.0,
                    f"ndcg@{k}": 0.0, f"hit@{k}": 0.0}
    return metrics | {"mrr": 0.0}


def evaluate(pipeline: RAGPipeline, golden: list[GoldenQuestion], *,
             use_judge: bool = False, workers: int = 1) -> dict[str, Any]:
    chunks = list(pipeline.index.chunks.values())

    def run_one(item: GoldenQuestion) -> QuestionResult:
        phrase = item.evidence_phrase.lower()
        total_relevant = sum(1 for c in chunks if phrase and phrase in c.text.lower())

        started = time.perf_counter()
        try:
            answer = pipeline.answer(item.question, use_cache=False)
        except RagError as exc:
            # One unlucky question must not destroy a run that costs money and
            # minutes. Record it, keep going, and report the failure count.
            return QuestionResult(
                id=item.id, question=item.question, expect_refusal=item.expect_refusal,
                refused=False, refusal_reason=None, retrieval=_empty_retrieval(),
                best_dense=-1.0, best_sparse=0.0, n_relevant_in_corpus=total_relevant,
                answer="", citations=[], groundedness=None, keyword_coverage=0.0,
                judge_score=None, latency_ms=round((time.perf_counter() - started) * 1000, 1),
                error=f"{type(exc).__name__}: {exc}"[:300])
        latency_ms = (time.perf_counter() - started) * 1000

        relevance = [bool(phrase) and phrase in c.chunk.text.lower() for c in answer.sources]
        retrieval = {}
        for k in K_VALUES:
            retrieval[f"recall@{k}"] = recall_at_k(relevance, k, total_relevant)
            retrieval[f"precision@{k}"] = precision_at_k(relevance, k)
            retrieval[f"ndcg@{k}"] = ndcg_at_k(relevance, k, total_relevant)
            retrieval[f"hit@{k}"] = hit_rate(relevance, k)
        retrieval["mrr"] = mrr(relevance)

        judge = None
        if use_judge and not answer.refused and answer.sources:
            judge = judge_faithfulness(pipeline.llm, item.question, answer.text,
                                       [c.chunk.text for c in answer.sources])

        return QuestionResult(
            id=item.id, question=item.question, expect_refusal=item.expect_refusal,
            refused=answer.refused, refusal_reason=answer.refusal_reason, retrieval=retrieval,
            best_dense=max((c.dense_score or -1.0) for c in answer.sources) if answer.sources else -1.0,
            best_sparse=max((c.sparse_score or 0.0) for c in answer.sources) if answer.sources else 0.0,
            n_relevant_in_corpus=total_relevant, answer=answer.text, citations=answer.citations,
            groundedness=answer.groundedness,
            keyword_coverage=_keyword_coverage(answer.text, item.must_include),
            judge_score=judge, latency_ms=round(latency_ms, 1),
            timings_ms=answer.timings_ms,
            sources=[c.explain() for c in answer.sources])

    # Threads, not async: the SDK call is blocking IO, so a small pool turns a
    # serial 13 minute run into a 3 minute one. Retrieval is read only here and
    # the cache is bypassed, so there is nothing to race on.
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_one, golden))
    else:
        results = [run_one(item) for item in golden]

    scored = [r for r in results if r.error is None]
    answerable = [r for r in scored if not r.expect_refusal]
    refusable = [r for r in scored if r.expect_refusal]
    latencies = [r.latency_ms for r in scored]

    aggregates: dict[str, Any] = {"questions": len(results), "answerable": len(answerable),
                                  "out_of_scope": len(refusable),
                                  "errors": len(results) - len(scored)}
    for key in _empty_retrieval():
        aggregates[key] = mean([r.retrieval[key] for r in answerable])
    stage_totals: dict[str, list[float]] = {}
    for result in scored:
        for stage, value in result.timings_ms.items():
            stage_totals.setdefault(stage, []).append(value)
    aggregates["stage_ms"] = {stage: mean(values) for stage, values in stage_totals.items()}
    aggregates.update({
        "answer_rate": mean([0.0 if r.refused else 1.0 for r in answerable]),
        "keyword_coverage": mean([r.keyword_coverage for r in answerable]),
        "groundedness": mean([r.groundedness or 0.0 for r in answerable if not r.refused]),
        "citation_rate": mean([1.0 if r.citations else 0.0 for r in answerable if not r.refused]),
        "correct_refusals": mean([1.0 if r.refused else 0.0 for r in refusable]),
        "false_refusals": mean([1.0 if r.refused else 0.0 for r in answerable]),
        "latency_p50_ms": percentile(latencies, 50), "latency_p95_ms": percentile(latencies, 95),
    })
    if any(r.judge_score is not None for r in results):
        aggregates["judge_faithfulness"] = mean(
            [r.judge_score for r in results if r.judge_score is not None])

    # Threshold calibration: where to put min_dense_score so that in scope
    # questions pass and out of scope questions are refused.
    in_scope_scores = [r.best_dense for r in answerable]
    out_scores = [r.best_dense for r in refusable]
    calibration = {
        "in_scope_dense_p10": percentile(in_scope_scores, 10),
        "in_scope_dense_p50": percentile(in_scope_scores, 50),
        "out_of_scope_dense_p90": percentile(out_scores, 90),
        "suggested_min_dense_score": round(
            (percentile(in_scope_scores, 10) + percentile(out_scores, 90)) / 2, 3)
        if in_scope_scores and out_scores else None,
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": pipeline.settings.fingerprint(),
        "index": pipeline.index.stats(),
        "aggregates": aggregates, "calibration": calibration,
        "results": [asdict(r) for r in results],
    }


def format_report(report: dict[str, Any]) -> str:
    lines = ["", f"Index: {report['index']['chunks']} chunks, version {report['index']['index_version']}",
             f"Config: {json.dumps(report['config'])}", "", "AGGREGATES"]
    for key, value in report["aggregates"].items():
        lines.append(f"  {key:24} {value}")
    lines += ["", "CALIBRATION (for settings.min_dense_score)"]
    for key, value in report["calibration"].items():
        lines.append(f"  {key:28} {value}")
    lines += ["", f"{'id':5} {'ref':4} {'rec@5':6} {'mrr':5} {'grnd':5} {'kw':5}  question"]
    for row in report["results"]:
        if row.get("error"):
            lines.append(f"{row['id']:5} ERROR                             {row['error'][:70]}")
            continue
        lines.append(
            f"{row['id']:5} {str(row['refused'])[:4]:4} {row['retrieval']['recall@5']:<6.2f} "
            f"{row['retrieval']['mrr']:<5.2f} {(row['groundedness'] or 0):<5.2f} "
            f"{row['keyword_coverage']:<5.2f}  {row['question'][:60]}")
    return "\n".join(lines)
