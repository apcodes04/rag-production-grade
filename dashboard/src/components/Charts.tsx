import {
  Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";

import type { Aggregates, Calibration, QuestionResult } from "../types";
import { useChartColors } from "./chartColors";
import { Card } from "./ui";

/** Recharts needs real colour strings, not CSS variables, so every chart reads
 *  the current token values and re-reads them when the theme changes. Hard
 *  coded hex would stay dark on a light page. */
function useChartTheme() {
  const c = useChartColors();
  return {
    c,
    axis: { stroke: c.axis, fontSize: 13 },
    tooltip: {
      contentStyle: {
        background: c.card, border: `1px solid ${c.line}`, borderRadius: 12,
        color: c.ink, fontSize: 13,
      },
    },
    legend: { wrapperStyle: { fontSize: 13, color: c.axis } },
  };
}

export function RetrievalChart({ aggregates }: { aggregates: Aggregates }) {
  const { c, axis, tooltip, legend } = useChartTheme();
  const data = [1, 3, 5, 10].map((k) => ({
    k: `@${k}`,
    recall: Number(aggregates[`recall@${k}`] ?? 0),
    precision: Number(aggregates[`precision@${k}`] ?? 0),
    ndcg: Number(aggregates[`ndcg@${k}`] ?? 0),
  }));
  return (
    <Card title="Retrieval quality"
          hint="Recall caps everything downstream: the generator cannot use a chunk retrieval never found.">
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
            <CartesianGrid stroke={c.grid} vertical={false} />
            <XAxis dataKey="k" tick={axis} />
            <YAxis domain={[0, 1]} tick={axis} />
            <Tooltip {...tooltip} />
            <Legend {...legend} />
            <Bar dataKey="recall" fill={c.c1} radius={[6, 6, 0, 0]} />
            <Bar dataKey="precision" fill={c.c2} radius={[6, 6, 0, 0]} />
            <Bar dataKey="ndcg" fill={c.c3} radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-3 text-sm text-ink-soft">
        Precision falls as k grows because each question has one relevant chunk,
        so most slots are noise by construction.
      </p>
    </Card>
  );
}

export function LatencyChart({ aggregates }: { aggregates: Aggregates }) {
  const { c, axis, tooltip } = useChartTheme();
  const stages = aggregates.stage_ms ?? {};
  const data = Object.entries(stages).map(([stage, ms]) => ({ stage, ms: Number(ms) }));
  return (
    <Card title="Where the time goes"
          hint="Average milliseconds per stage. The LLM call and the query embedding are the whole bill; everything local is free.">
      {data.length === 0 ? (
        <p className="text-base text-ink-soft">
          This run predates per stage timings. Re-run{" "}
          <code className="font-mono">evaluate.py</code> to record them.
        </p>
      ) : (
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical"
                      margin={{ top: 8, right: 16, left: 24, bottom: 4 }}>
              <CartesianGrid stroke={c.grid} horizontal={false} />
              <XAxis type="number" tick={axis} unit=" ms" />
              <YAxis type="category" dataKey="stage" tick={axis} width={100} />
              <Tooltip {...tooltip}
                       formatter={(v: number) => [`${v.toFixed(1)} ms`, "average"]} />
              <Bar dataKey="ms" radius={[0, 6, 6, 0]}>
                {data.map((row) => (
                  <Cell key={row.stage}
                        fill={row.ms > 200 ? c.ember : row.ms > 20 ? c.c2 : c.c1} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}

/** The most honest chart here: every question's best dense score, split by
 *  whether the corpus can answer it. If the two clouds overlap, no threshold
 *  separates them, and saying so out loud is the point. */
export function CalibrationChart({ results, calibration }: {
  results: QuestionResult[]; calibration: Calibration;
}) {
  const { c, axis, tooltip, legend } = useChartTheme();
  const inScope = results.filter((r) => !r.expect_refusal && !r.error)
    .map((r, i) => ({ x: i + 1, y: r.best_dense, id: r.id }));
  const outScope = results.filter((r) => r.expect_refusal && !r.error)
    .map((r, i) => ({ x: i + 1, y: r.best_dense, id: r.id }));
  const threshold = calibration.suggested_min_dense_score;
  const margin = calibration.in_scope_dense_p10 - calibration.out_of_scope_dense_p90;

  return (
    <Card title="Refusal threshold calibration"
          hint="Each dot is one question's best dense similarity. Green can be answered from the documents, red cannot.">
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid stroke={c.grid} />
            <XAxis type="number" dataKey="x" tick={axis} name="question" />
            <YAxis type="number" dataKey="y" domain={[0, 1]} tick={axis} name="best dense score" />
            <Tooltip {...tooltip}
                     formatter={(value: number, name: string) =>
                       [name === "y" ? value.toFixed(3) : value, name]}
                     labelFormatter={() => ""} />
            <Legend {...legend} />
            {threshold !== null && (
              <ReferenceLine y={threshold} stroke={c.spot} strokeDasharray="4 4"
                             label={{ value: `suggested ${threshold}`, fill: c.spot,
                                      fontSize: 12, position: "insideTopRight" }} />
            )}
            <Scatter name="in scope" data={inScope} fill={c.c1} />
            <Scatter name="out of scope" data={outScope} fill={c.ember} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-3 text-base leading-relaxed text-ink-soft">
        In scope p10 <span className="font-mono text-ink">{calibration.in_scope_dense_p10}</span>,
        out of scope p90 <span className="font-mono text-ink">{calibration.out_of_scope_dense_p90}</span>,
        margin <span className={`font-mono font-bold ${margin > 0.1 ? "text-sel" : "text-ember"}`}>
          {margin.toFixed(3)}
        </span>.{" "}
        {margin > 0.1
          ? "The two groups separate cleanly, so a similarity threshold can refuse out of scope questions before the LLM is ever called."
          : "The two groups overlap, so no single threshold separates them on this corpus. Refusals here come from the prompt and the output guard, not from the retrieval gate."}
      </p>
    </Card>
  );
}
