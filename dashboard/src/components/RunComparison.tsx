import type { RunSummary } from "../types";
import { Card, Pill } from "./ui";

const COLUMNS: Array<[string, (r: RunSummary) => number | string, "high" | "low" | "none"]> = [
  ["recall@1", (r) => Number(r.aggregates["recall@1"] ?? 0), "high"],
  ["recall@3", (r) => Number(r.aggregates["recall@3"] ?? 0), "high"],
  ["MRR", (r) => r.aggregates.mrr, "high"],
  ["grounded", (r) => r.aggregates.groundedness, "high"],
  ["citations", (r) => r.aggregates.citation_rate, "high"],
  ["correct refusals", (r) => r.aggregates.correct_refusals, "high"],
  ["false refusals", (r) => r.aggregates.false_refusals, "low"],
  ["p50 ms", (r) => r.aggregates.latency_p50_ms, "low"],
  ["p95 ms", (r) => r.aggregates.latency_p95_ms, "low"],
  ["errors", (r) => r.aggregates.errors, "low"],
];

export function RunComparison({ runs, selected, onSelect }: {
  runs: RunSummary[]; selected: string; onSelect: (id: string) => void;
}) {
  const best = new Map<string, number>();
  for (const [label, read, direction] of COLUMNS) {
    if (direction === "none") continue;
    const values = runs.map((run) => Number(read(run)));
    best.set(label, direction === "high" ? Math.max(...values) : Math.min(...values));
  }

  return (
    <Card title="Every evaluation run"
          hint="One row per configuration, newest first. A change that is not in this table did not happen.">
      <div className="scroll-x">
        <table className="w-full min-w-[1100px] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs tracking-wide text-ink-soft">
              <th className="py-2 pr-4 font-medium">run</th>
              <th className="py-2 pr-4 font-medium">questions</th>
              {COLUMNS.map(([label]) => (
                <th key={label} className="py-2 pr-4 font-medium">{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id}
                  onClick={() => onSelect(run.id)}
                  className={`cursor-pointer border-b transition-colors hover:bg-panel ${
                    run.id === selected ? "border-lime bg-panel" : "border-line"
                  }`}>
                <td className="py-3 pr-4">
                  <div className="flex items-center gap-2">
                    {run.id === selected && <Pill tone="good">viewing</Pill>}
                    <span className="font-mono text-xs text-ink">{run.id}</span>
                  </div>
                  <div className="mt-1 text-xs text-ink-soft">
                    {String(run.config.llm_model)} | chunk {String(run.config.chunk_size)} |
                    {" "}reranker {String(run.config.reranker)} | gate {String(run.config.min_dense_score)}
                  </div>
                </td>
                <td className="py-3 pr-4 tabular-nums text-ink-soft">
                  {run.aggregates.questions}
                  <span className="text-xs"> ({run.aggregates.out_of_scope} out of scope)</span>
                </td>
                {COLUMNS.map(([label, read]) => {
                  const value = Number(read(run));
                  const isBest = runs.length > 1 && value === best.get(label);
                  const digits = label.includes("ms") || label === "errors" ? 0 : 3;
                  return (
                    <td key={label}
                        className={`py-3 pr-4 tabular-nums ${isBest ? "font-semibold text-sel" : "text-ink"}`}>
                      {value.toFixed(digits)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-ink-soft">
        Green marks the best value in each column. Click any row to load it above.
        Latency columns are the honest cost; a run that is fast because it refused
        everything would show it in the refusal columns.
      </p>
    </Card>
  );
}
