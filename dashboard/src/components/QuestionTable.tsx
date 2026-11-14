import { Fragment, useEffect, useRef, useState } from "react";

import type { QuestionResult } from "../types";
import { Card, Pill, fmt } from "./ui";

function Verdict({ row }: { row: QuestionResult }) {
  if (row.error) return <Pill tone="bad">error</Pill>;
  if (row.expect_refusal) {
    return row.refused ? <Pill tone="good">refused correctly</Pill> : <Pill tone="bad">should have refused</Pill>;
  }
  return row.refused ? <Pill tone="bad">refused a real question</Pill> : <Pill tone="good">answered</Pill>;
}

export function QuestionTable({ results }: { results: QuestionResult[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  // id and question are the columns a reader actually wants first, so a run
  // switch should never leave the table scrolled past them from before.
  useEffect(() => { if (scroller.current) scroller.current.scrollLeft = 0; }, [results]);

  return (
    <Card title="Every question in the run"
          hint="Click a row to see the answer, the chunks that were retrieved and every score they collected on the way.">
      <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-ink-soft">
        <span aria-hidden>↔</span> Scroll sideways to see every column, including id and question
      </p>
      <div className="relative">
        <div ref={scroller} className="scroll-x">
        <table className="w-full min-w-[900px] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs tracking-wide text-ink-soft">
              <th className="py-2 pr-4 font-medium">id</th>
              <th className="py-2 pr-4 font-medium">question</th>
              <th className="py-2 pr-4 font-medium">verdict</th>
              <th className="py-2 pr-4 font-medium">recall@5</th>
              <th className="py-2 pr-4 font-medium">MRR</th>
              <th className="py-2 pr-4 font-medium">grounded</th>
              <th className="py-2 pr-4 font-medium">best dense</th>
              <th className="py-2 pr-4 font-medium">ms</th>
            </tr>
          </thead>
          <tbody>
            {results.map((row) => (
              <Fragment key={row.id}>
                <tr
                    onClick={() => setOpen(open === row.id ? null : row.id)}
                    className="cursor-pointer border-b border-line hover:bg-panel">
                  <td className="py-2.5 pr-4 font-mono text-xs text-ink-soft">{row.id}</td>
                  <td className="max-w-md truncate py-2.5 pr-4 text-ink">{row.question}</td>
                  <td className="py-2.5 pr-4"><Verdict row={row} /></td>
                  <td className="py-2.5 pr-4 tabular-nums">{fmt(row.retrieval["recall@5"], 2)}</td>
                  <td className="py-2.5 pr-4 tabular-nums">{fmt(row.retrieval.mrr, 2)}</td>
                  <td className="py-2.5 pr-4 tabular-nums">{fmt(row.groundedness, 2)}</td>
                  <td className="py-2.5 pr-4 tabular-nums">{fmt(row.best_dense, 3)}</td>
                  <td className="py-2.5 pr-4 tabular-nums text-ink-soft">{row.latency_ms.toFixed(0)}</td>
                </tr>
                {open === row.id && (
                  <tr className="border-b border-line bg-panel">
                    <td colSpan={8} className="p-5">
                      <div className="grid gap-5 lg:grid-cols-2">
                        <div>
                          <h4 className="mb-2 text-xs tracking-wide text-ink-soft">answer</h4>
                          <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink">
                            {row.error ?? row.answer}
                          </p>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {(row.citations ?? []).map((c) => <Pill key={c}>cited [{c}]</Pill>)}
                            {row.refusal_reason && <Pill tone="warn">{row.refusal_reason}</Pill>}
                          </div>
                          {Object.keys(row.timings_ms ?? {}).length > 0 && (
                            <div className="mt-3 font-mono text-xs text-ink-soft">
                              {Object.entries(row.timings_ms ?? {})
                                .map(([stage, ms]) => `${stage} ${ms}ms`).join("  |  ")}
                            </div>
                          )}
                        </div>
                        <div>
                          <h4 className="mb-2 text-xs tracking-wide text-ink-soft">
                            retrieved chunks, in the order the LLM saw them
                          </h4>
                          <table className="w-full text-xs">
                            <thead className="text-ink-soft">
                              <tr>
                                <th className="pb-1 text-left font-medium">#</th>
                                <th className="pb-1 text-left font-medium">source</th>
                                <th className="pb-1 text-left font-medium">dense</th>
                                <th className="pb-1 text-left font-medium">sparse</th>
                                <th className="pb-1 text-left font-medium">fused</th>
                              </tr>
                            </thead>
                            <tbody>
                              {(row.sources ?? []).map((source, i) => (
                                <tr key={source.chunk_id} className="border-t border-line">
                                  <td className="py-1 text-ink-soft">[{i + 1}]</td>
                                  <td className="py-1">
                                    {source.source}{source.page ? ` p${source.page}` : ""}
                                  </td>
                                  <td className="py-1 tabular-nums">{fmt(source.dense)}</td>
                                  <td className="py-1 tabular-nums">{fmt(source.sparse)}</td>
                                  <td className="py-1 tabular-nums">{fmt(source.fused, 4)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {(row.sources ?? []).length === 0 && (
                            <p className="text-xs text-ink-soft">
                              No sources recorded (older report format, or the query was blocked
                              before retrieval).
                            </p>
                          )}
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
        </div>
        {/* A fixed cue that there is more table past the edge, since a plain
         *  unstyled horizontal scrollbar is easy to miss on a touch screen. */}
        <div aria-hidden className="pointer-events-none absolute inset-y-0 right-0 w-10 bg-gradient-to-l from-card to-transparent" />
      </div>
    </Card>
  );
}
