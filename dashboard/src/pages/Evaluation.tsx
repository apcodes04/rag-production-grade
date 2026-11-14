import { useEffect, useState } from "react";

import { api } from "../api";
import { CalibrationChart, LatencyChart, RetrievalChart } from "../components/Charts";
import { QuestionTable } from "../components/QuestionTable";
import { RunComparison } from "../components/RunComparison";
import { Card, Disclosure, ErrorBox, Loading, Stat, fmt, pct } from "../components/ui";
import type { RunDetail, RunSummary } from "../types";

export function EvaluationPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [selected, setSelected] = useState<string>("");
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    void api.runs().then((list) => {
      setRuns(list);
      if (list.length) setSelected(list[0].id);
    }).catch((exc) => setError(exc as Error));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setDetail(null);
    void api.run(selected).then(setDetail).catch((exc) => setError(exc as Error));
  }, [selected]);

  const aggregates = detail?.aggregates;

  return (
    <>
      <Card title="Measured quality" tone="violet"
            hint={<>
              Recorded runs against one document: AI_QnA_Guide_Extended.pdf.{" "}
              <strong className="rounded bg-amber px-1.5 py-0.5 font-bold text-on-amber">
                Not your uploaded document.
              </strong>
            </>}>
        <p className="text-base leading-relaxed text-on-violet/80">
          Evaluation needs ground truth, meaning somebody has to label which
          chunk answers each question. I labelled 21 questions against that one
          document, so these are real measurements replayed from JSON in the
          repository. Nobody has labelled your upload, so scoring it would be
          invented, and this page does not pretend otherwise. The upload is
          exercised live on the other two pages instead.
        </p>
        <Disclosure summary="How the question set is built" tone="dark">
          Twenty one questions: eleven the documents answer, and ten they do
          not, including near misses chosen to sit close to the relevance
          threshold. Measuring refusals needs questions that deserve one, and
          easy negatives make any system look perfect.
        </Disclosure>
      </Card>

      {error && <ErrorBox error={error} />}
      {!runs && !error && <Loading what="evaluation runs" />}

      {runs && runs.length === 0 && (
        <Card title="No evaluation runs yet">
          <p className="text-base text-ink-soft">
            Run <code className="font-mono">python scripts/evaluate.py --tag baseline</code>{" "}
            and reload.
          </p>
        </Card>
      )}

      {detail && aggregates && (
        <>
          <Card title={detail.id ?? selected} tone="violet"
                hint={`${aggregates.questions} questions, ${aggregates.out_of_scope} of them deliberately out of scope. ${String(detail.config.llm_model)}, chunk ${String(detail.config.chunk_size)}, reranker ${String(detail.config.reranker)}.`}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="recall@3" value={fmt(Number(aggregates["recall@3"]), 3)}
                    good={Number(aggregates["recall@3"]) >= 0.9}
                    explain="Relevant chunks found in the top 3." />
              <Stat label="MRR" value={fmt(aggregates.mrr, 3)} good={aggregates.mrr >= 0.8}
                    explain="How high the first relevant chunk ranks." />
              <Stat label="groundedness" value={fmt(aggregates.groundedness, 3)}
                    good={aggregates.groundedness >= 0.7}
                    explain="Answer content that appears in the cited chunks." />
              <Stat label="citation rate" value={pct(aggregates.citation_rate)}
                    good={aggregates.citation_rate >= 0.95}
                    explain="Answers citing at least one real source." />
              <Stat label="correct refusals" value={pct(aggregates.correct_refusals)}
                    good={aggregates.correct_refusals >= 0.95}
                    explain="Out of scope questions declined." />
              <Stat label="false refusals" value={pct(aggregates.false_refusals)}
                    good={aggregates.false_refusals === 0}
                    explain="Real questions wrongly refused. Must stay at zero." />
              <Stat label="latency p50" value={aggregates.latency_p50_ms.toFixed(0)} suffix="ms"
                    good={aggregates.latency_p50_ms < 3000}
                    explain="Median end to end, including the LLM call." />
              <Stat label="latency p95" value={aggregates.latency_p95_ms.toFixed(0)} suffix="ms"
                    good={aggregates.latency_p95_ms < 5000}
                    explain="The slow tail. A gap from p50 means retries, not slow code." />
            </div>

            {aggregates.errors > 0 && (
              <p className="mt-4 text-base font-semibold text-ember">
                {aggregates.errors} question(s) failed with a provider error and are
                excluded from the averages above.
              </p>
            )}

            <Disclosure summary="How to read these together" tone="dark">
              Read recall first, because the generator can only use what
              retrieval found. Then read the two refusal numbers as a pair:
              correct refusals should be high and false refusals zero, and
              either one alone can be gamed by a system that always answers or
              never does.
            </Disclosure>
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <RetrievalChart aggregates={aggregates} />
            <LatencyChart aggregates={aggregates} />
          </div>

          <CalibrationChart results={detail.results} calibration={detail.calibration} />
        </>
      )}

      {runs && runs.length > 0 && (
        <RunComparison runs={runs} selected={selected} onSelect={setSelected} />
      )}

      {detail && <QuestionTable results={detail.results} />}
      {selected && !detail && !error && <Loading what="run detail" />}
    </>
  );
}
