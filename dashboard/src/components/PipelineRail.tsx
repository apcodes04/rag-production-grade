import { useState } from "react";

import { Disclosure } from "./ui";
import type { AnswerResponse, IndexStats } from "../types";

/** The stages the pipeline reports, in execution order. Keys match the
 *  _stage() calls in pipeline.py, so a stage renamed in the backend shows up
 *  here as missing rather than as a wrong label. */
export const STAGES: { key: string; title: string; why: string }[] = [
  {
    key: "input_guard", title: "Input guard",
    why: "Length, injection patterns, and masking anything that looks personal before it reaches the logs. The cheapest place to stop a bad request is before it costs an embedding call.",
  },
  {
    key: "rewrite", title: "Query rewrite",
    why: "Retrieval has no memory, so a follow up like \"and the second one?\" is turned into a standalone question first. Skipped when there is no history.",
  },
  {
    key: "cache_lookup", title: "Cache lookup",
    why: "Keyed on the question, the history, the index version, the prompt version, the config and the chosen documents. Key on the question alone and you serve yesterday's answer after re-indexing.",
  },
  {
    key: "retrieval", title: "Hybrid retrieval",
    why: "Dense vectors find meaning and miss exact identifiers. BM25 finds exact terms and misses paraphrase. Their ranks are fused, which needs no calibration between two incomparable scales.",
  },
  {
    key: "rerank", title: "Rerank",
    why: "A cross encoder reads query and chunk together. More accurate, much slower, so it runs on a short list or not at all. Off by default here.",
  },
  {
    key: "generate", title: "Generation",
    why: "Chunks are tagged as untrusted data, which is what stops text inside a document acting as an instruction. The call carries a retry, a deadline, a rate limiter and a fallback model.",
  },
  {
    key: "output_guard", title: "Output guard",
    why: "Checks every citation points at a real retrieved chunk, and measures how much of the answer appears in those chunks. The last chance to catch an answer that reads well and is not supported.",
  },
];

/** What happens inside, beside the conversation rather than on a page of its
 *  own. It fills in as soon as an answer lands, so the stages are read in the
 *  same glance as the answer they produced. */
export function PipelineRail({ answer, health, busy }: {
  answer: AnswerResponse | null; health: IndexStats | null; busy: boolean;
}) {
  const timings = answer?.timings_ms ?? {};
  const total = Object.values(timings).reduce((a, b) => a + b, 0);
  const slowest = Object.entries(timings).sort((a, b) => b[1] - a[1])[0]?.[0];

  return (
    <aside className="space-y-4 lg:min-h-0 lg:overflow-y-auto lg:pr-1">
      <div className="surface rounded-3xl border border-line bg-card p-5">
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="text-lg font-bold tracking-tight text-ink">What happens inside</h2>
          {busy
            ? <span className="flex items-center gap-1.5 text-sm font-semibold text-sel">
                <span className="size-2 animate-pulse rounded-full bg-sel" />running
              </span>
            : total > 0 && (
                <span className="font-mono text-sm font-bold text-ink">{total.toFixed(0)} ms</span>
              )}
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2">
          {STAGES.map((stage, i) => {
            const ms = timings[stage.key];
            const ran = ms !== undefined;
            const share = ran && total > 0 ? (ms / total) * 100 : 0;
            return (
              <div key={stage.key}
                   className={`rounded-xl border p-2.5 ${
                     stage.key === slowest
                       ? "border-lime bg-lime/15"
                       : ran ? "border-line bg-panel" : "border-line/60"}`}>
                <div className="flex items-baseline gap-1.5">
                  <span className="font-mono text-[11px] text-ink-soft">{i + 1}</span>
                  <span className={`text-xs font-bold leading-tight ${
                    ran ? "text-ink" : "text-ink-soft"}`}>
                    {stage.title}
                  </span>
                </div>
                <div className="mt-1 font-mono text-sm tabular-nums">
                  {ran
                    ? <span className="font-bold text-sel">{ms.toFixed(0)}<span className="text-xs"> ms</span></span>
                    : <span className="text-xs text-ink-soft">
                        {busy ? "running" : answer ? "skipped" : "idle"}
                      </span>}
                </div>
                {share > 0 && (
                  <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-card">
                    <div className="bar-grow h-full rounded-full bg-lime"
                         style={{ width: `${share}%` }} />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {!answer && !busy && (
          <p className="mt-4 text-sm text-ink-soft">
            Ask something and these fill with the real timings for it.
          </p>
        )}

        <Disclosure summary="Why each stage exists">
          <ul className="space-y-3 text-sm">
            {STAGES.map((stage) => (
              <li key={stage.key}>
                <span className="font-bold text-ink">{stage.title}. </span>{stage.why}
              </li>
            ))}
          </ul>
        </Disclosure>
      </div>

      {health && (
        <div className="surface rounded-3xl border border-amber-line bg-amber p-5 text-on-amber">
          <h3 className="text-sm font-bold tracking-wide">The index</h3>
          <dl className="mt-3 space-y-1.5 text-sm">
            {[
              ["documents", String(health.documents)],
              ["chunks", String(health.chunks)],
              ["vectors", String(health.vectors)],
              ["dimensions", String(health.dim)],
              ["model", health.embedding_model],
              ["version", health.index_version],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between gap-3">
                <dt className="opacity-70">{label}</dt>
                <dd className="truncate font-mono font-semibold">{value}</dd>
              </div>
            ))}
          </dl>
          <Disclosure summary="Why the version matters">
            It is a hash of the index contents and part of the cache key, so
            re-indexing invalidates exactly the answers that depended on the
            changed document. The hash comes from content, not file names, so
            renaming a file changes nothing.
          </Disclosure>
        </div>
      )}
    </aside>
  );
}


/** The same pipeline, as one bar for narrow screens.
 *
 *  Collapsed it shows the total and the slowest stage, which is the whole
 *  story in one line. Expanded it shows every stage. A full sidebar below the
 *  chat would be invisible on a phone, which is the same as not having it.
 */
export function PipelineStrip({ answer, health, busy, open, onToggle }: {
  answer: AnswerResponse | null; health: IndexStats | null;
  busy: boolean; open: boolean; onToggle: () => void;
}) {
  const timings = answer?.timings_ms ?? {};
  const slowest = Object.entries(timings).sort((a, b) => b[1] - a[1])[0];

  return (
    <div className={`surface overflow-hidden rounded-2xl border ${
      open ? "border-line bg-card" : "border-lime bg-lime/20"}`}>
      <button type="button" onClick={onToggle} aria-expanded={open}
              className="flex w-full items-center gap-2 px-4 py-2.5 text-left">
        <span className="text-sm font-bold text-ink">What happens inside</span>
        {busy && (
          <span className="ml-1 flex items-center gap-1 text-xs font-bold text-sel">
            <span className="size-1.5 animate-pulse rounded-full bg-sel" />running
          </span>
        )}
        <span className={`ml-auto shrink-0 text-sm font-bold ${
          open ? "text-ink-soft" : "text-ink"}`}>
          {open ? "hide" : "tap to open"}
          <span className={`ml-1 inline-block transition-transform ${open ? "rotate-90" : ""}`}>
            ›
          </span>
        </span>
      </button>

      {open && (
        <div className="border-t border-line px-2.5 py-1.5">
          <div className="divide-y divide-line/60">
            {STAGES.map((stage, i) => {
              const ms = timings[stage.key];
              const ran = ms !== undefined;
              return (
                <div key={stage.key}
                     className={`flex items-center gap-2 py-1.5 ${
                       stage.key === slowest?.[0] ? "bg-lime/10" : ""}`}>
                  <span className="font-mono text-[11px] text-ink-soft">{i + 1}</span>
                  <span className="flex-1 text-xs font-bold text-ink">{stage.title}</span>
                  {ran
                    ? <span className="shrink-0 font-mono text-xs font-bold tabular-nums text-sel">{ms.toFixed(0)} ms</span>
                    : <span className="shrink-0 text-[11px] text-ink-soft">{busy ? "running" : "skipped"}</span>}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}


/** The pipeline, as a slim rail beside the chat on narrow screens, so the
 *  chat gets the full height instead of sharing it with a block stacked on
 *  top. Tapping it opens the full breakdown as a sheet over the chat,
 *  rather than pushing the chat out of the way to show it inline. */
export function PipelineSideRail({ answer, health, busy, open, onToggle, scopeNames, scopeAll,
                                  documentCount, onChangeScope }: {
  answer: AnswerResponse | null; health: IndexStats | null;
  busy: boolean; open: boolean; onToggle: () => void;
  scopeNames: string[]; scopeAll: boolean; documentCount: number | null;
  onChangeScope: () => void;
}) {
  const timings = answer?.timings_ms ?? {};
  const total = Object.values(timings).reduce((a, b) => a + b, 0);
  const slowest = Object.entries(timings).sort((a, b) => b[1] - a[1])[0];
  const [namesOpen, setNamesOpen] = useState(false);
  const count = scopeNames.length || documentCount || 0;

  return (
    <>
      <button type="button" onClick={onToggle} aria-expanded={open}
              className="surface flex w-[4.75rem] shrink-0 flex-col items-center gap-1.5 rounded-2xl border border-lime bg-lime/20 px-1.5 py-3 text-center">
        <span className="text-[11px] font-bold leading-tight text-ink">Pipeline</span>
        {busy
          ? <span className="size-2 animate-pulse rounded-full bg-sel" />
          : total > 0
            ? <span className="font-mono text-xs font-bold leading-tight text-ink">
                {total.toFixed(0)}<br />ms
              </span>
            : <span className="text-[11px] text-ink-soft">idle</span>}
        <span className="text-[11px] font-semibold text-ink-soft">{count} doc{count === 1 ? "" : "s"}</span>
        <span className="mt-auto text-[11px] font-bold text-ink">tap ›</span>
      </button>

      {open && (
        <div className="fixed inset-0 z-30 flex items-end bg-ink/40" onClick={onToggle}>
          <div className="max-h-[82dvh] w-full space-y-3 overflow-y-auto rounded-t-3xl border-t border-line bg-card p-4"
               onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold tracking-tight text-ink">What happens inside</h2>
              <button onClick={onToggle} className="text-sm font-semibold text-ink-soft underline underline-offset-4">
                close
              </button>
            </div>

            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-xl border border-line bg-panel px-3 py-2 text-xs">
              <span className="shrink-0 text-ink-soft">asking</span>
              <button onClick={() => setNamesOpen(!namesOpen)}
                      className="shrink-0 rounded-full border border-line bg-card px-2 py-0.5 font-bold text-ink">
                {scopeAll ? `all ${count}` : count} doc{count === 1 ? "" : "s"} {namesOpen ? "▾" : "▸"}
              </button>
              {health && <span className="text-ink-soft">· {health.chunks} chunks</span>}
              <button onClick={onChangeScope}
                      className="ml-auto shrink-0 font-semibold text-ink-soft underline underline-offset-2">
                change
              </button>
            </div>

            {namesOpen && (
              <div className="flex flex-wrap gap-1.5">
                {scopeNames.length > 0
                  ? scopeNames.map((name) => (
                      <span key={name} className="truncate rounded-full border border-lime bg-lime/15 px-2 py-0.5 text-[11px] font-semibold text-ink">
                        {name}
                      </span>
                    ))
                  : <span className="text-[11px] text-ink-soft">no documents yet</span>}
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              {STAGES.map((stage, i) => {
                const ms = timings[stage.key];
                const ran = ms !== undefined;
                return (
                  <div key={stage.key}
                       className={`flex items-center justify-between gap-2 rounded-lg border px-2.5 py-2 ${
                         stage.key === slowest?.[0]
                           ? "border-lime bg-lime/15"
                           : ran ? "border-line bg-panel" : "border-line/60"}`}>
                    <span className="truncate text-xs font-bold text-ink">
                      <span className="font-mono text-[11px] text-ink-soft">{i + 1} </span>{stage.title}
                    </span>
                    {ran
                      ? <span className="shrink-0 font-mono text-xs font-bold tabular-nums text-sel">{ms.toFixed(0)}ms</span>
                      : <span className="shrink-0 text-[11px] text-ink-soft">{busy ? "..." : "skipped"}</span>}
                  </div>
                );
              })}
            </div>

            {health && (
              <div className="rounded-xl border border-amber-line bg-amber p-3 text-on-amber">
                <h3 className="text-xs font-bold tracking-wide">The index</h3>
                <dl className="mt-2 space-y-1 text-xs">
                  {[
                    ["documents", String(health.documents)],
                    ["chunks", String(health.chunks)],
                    ["vectors", String(health.vectors)],
                    ["dimensions", String(health.dim)],
                    ["version", health.index_version],
                  ].map(([label, value]) => (
                    <div key={label} className="flex justify-between gap-3">
                      <dt className="opacity-70">{label}</dt>
                      <dd className="truncate font-mono font-semibold">{value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}


/** The index facts, as one quiet line for the bottom of a phone screen. */
export function IndexLine({ health }: { health: IndexStats | null }) {
  if (!health) return null;
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-0.5 rounded-xl border border-amber-line bg-amber px-3 py-1.5 text-[11px] text-on-amber">
      <span><span className="opacity-70">documents</span> <b>{health.documents}</b></span>
      <span><span className="opacity-70">chunks</span> <b>{health.chunks}</b></span>
      <span><span className="opacity-70">dims</span> <b>{health.dim}</b></span>
      <span className="truncate"><span className="opacity-70">index</span>{" "}
        <b className="font-mono">{health.index_version}</b></span>
    </div>
  );
}
