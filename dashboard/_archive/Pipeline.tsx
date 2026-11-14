import { Card, Cta, Disclosure, Pill } from "../components/ui";
import type { AnswerResponse, IndexStats } from "../types";

/** The stages the pipeline reports, in execution order. Keys match the
 *  _stage() calls in pipeline.py, so a stage renamed in the backend shows up
 *  here as missing rather than as a wrong label. */
const STAGES: { key: string; title: string; what: string; why: string }[] = [
  {
    key: "input_guard", title: "Input guardrail",
    what: "Checks length, scans for injection patterns, masks anything that looks like personal data before it reaches the logs.",
    why: "The cheapest place to stop a bad request is before it costs an embedding call. Masking is for logs only, so the question itself is never altered.",
  },
  {
    key: "rewrite", title: "Query rewrite",
    what: "Turns a follow up like \"and the second one?\" into a standalone question.",
    why: "Retrieval has no memory. An ambiguous follow up embeds to the wrong region of vector space, so it is resolved first. Skipped when there is no history.",
  },
  {
    key: "cache_lookup", title: "Cache lookup",
    what: "Looks for this question in SQLite, keyed on the question, the history, the index version, the prompt version and a config fingerprint.",
    why: "All five parts matter. Key on the question alone and you serve yesterday's answer after re-indexing, which is the worst kind of bug: silent and plausible.",
  },
  {
    key: "retrieval", title: "Hybrid retrieval",
    what: "Runs dense vector search and BM25 keyword search, then fuses the two ranked lists with reciprocal rank fusion.",
    why: "Dense finds meaning and misses exact identifiers. BM25 finds exact terms and misses paraphrase. Fusing by rank needs no score calibration between two incomparable scales.",
  },
  {
    key: "rerank", title: "Rerank",
    what: "Optionally re-scores the candidates with a cross encoder that reads query and chunk together.",
    why: "Fusion ranks by position, not by reading. A cross encoder is more accurate and much slower, so it runs on a short list or not at all. Off by default here.",
  },
  {
    key: "generate", title: "Generation",
    what: "Builds the prompt with the chunks tagged as untrusted data, then calls the model with a retry, a deadline, a rate limiter and a fallback model.",
    why: "Tagging the chunks is what stops text inside a document acting as an instruction. The deadline matters more than the retry count: ten retries that each wait a minute is an outage, not resilience.",
  },
  {
    key: "output_guard", title: "Output guardrail",
    what: "Verifies every citation points at a real retrieved chunk, and measures how much of the answer appears in those chunks.",
    why: "A model can cite source [4] when only three were retrieved. This is the last chance to catch an answer that reads perfectly and is not supported.",
  },
];

export function PipelinePage({ answer, health, goToAsk }: {
  answer: AnswerResponse | null; health: IndexStats | null; goToAsk: () => void;
}) {
  const timings = answer?.timings_ms ?? {};
  const total = Object.values(timings).reduce((a, b) => a + b, 0);
  const slowest = Object.entries(timings).sort((a, b) => b[1] - a[1])[0]?.[0];

  return (
    <>
      <Card title="How an answer is produced"
            hint={answer
              ? `Timings below are from your last question: "${answer.question}".`
              : "Ask a question and the real timings for it appear here."}>
        {answer ? (
          <div className="flex flex-wrap items-center gap-2">
            <Pill tone="lime">{total.toFixed(0)} ms total</Pill>
            {answer.cache_hit && <Pill tone="good">served from cache</Pill>}
            {slowest && <Pill>slowest: {slowest}</Pill>}
            <Pill>{answer.usage?.prompt_tokens ?? 0} tokens in</Pill>
            <Pill>{answer.usage?.output_tokens ?? 0} out</Pill>
          </div>
        ) : (
          <Cta onClick={goToAsk} pulse>Go ask a question</Cta>
        )}
      </Card>

      {STAGES.map((stage, i) => {
        const ms = timings[stage.key];
        const ran = ms !== undefined;
        const share = ran && total > 0 ? (ms / total) * 100 : 0;
        return (
          <section key={stage.key} data-rv="up"
                   className={`rounded-3xl border bg-card p-6 ${
                     stage.key === slowest ? "border-lime" : "border-line"}`}>
            <div className="flex flex-wrap items-baseline gap-3">
              <span className="font-mono text-sm text-ink-soft">{i + 1}</span>
              <h3 className="text-xl font-bold tracking-tight text-ink">{stage.title}</h3>
              {ran
                ? <span className="ml-auto font-mono text-lg font-bold tabular-nums text-sel">
                    {ms.toFixed(1)} ms
                  </span>
                : <span className="ml-auto text-sm text-ink-soft">
                    {answer ? "skipped here" : "not run yet"}
                  </span>}
            </div>

            {share > 0 && (
              <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-panel">
                <div className="bar-grow h-full rounded-full bg-lime" style={{ width: `${share}%` }} />
              </div>
            )}

            <p className="mt-3 text-base leading-relaxed text-ink">{stage.what}</p>
            <Disclosure summary="Why it exists">{stage.why}</Disclosure>
          </section>
        );
      })}

      {health && (
        <Card title="What the index holds">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["documents", String(health.documents)],
              ["chunks", String(health.chunks)],
              ["dimensions", String(health.dim)],
              ["index version", health.index_version],
            ].map(([label, value]) => (
              <div key={label} className="rounded-2xl border border-line bg-panel p-4">
                <div className="text-xs font-bold uppercase tracking-wide text-ink-soft">{label}</div>
                <div className="mt-1 truncate font-mono text-base text-ink">{value}</div>
              </div>
            ))}
          </div>
          <Disclosure summary="Why the index has a version">
            It is a hash of the index contents, and it is part of the cache key.
            That is how re-indexing a document invalidates exactly the answers
            that depended on it, instead of either serving stale text or
            throwing the whole cache away. Because the hash comes from content
            rather than file names, renaming a file changes nothing.
          </Disclosure>
        </Card>
      )}
    </>
  );
}
