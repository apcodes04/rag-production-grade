import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../api";
import { Card, Cta, Disclosure, ErrorBox, Ghost, Loading, Pill } from "../components/ui";
import type { DocumentsResponse, UploadResult } from "../types";

const size = (bytes: number) =>
  bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;

export function DocumentsPage({ onIndexChanged, goToAsk, selected, setSelected }: {
  onIndexChanged: () => void;
  goToAsk: () => void;
  selected: string[];
  setSelected: (paths: string[]) => void;
}) {
  const [data, setData] = useState<DocumentsResponse | null>(null);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setData(await api.documents());
    } catch (exc) {
      setError(exc as Error);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  async function send(files: FileList | null) {
    if (!files?.length) return;
    setError(null);
    // One at a time on purpose: each upload embeds its chunks, and firing
    // several in parallel is the quickest way to hit the rate limit.
    for (const file of Array.from(files)) {
      setBusy(file.name);
      try {
        const uploaded = await api.upload(file);
        setResult(uploaded);
        // A 200 response doesn't mean the file is now a searchable document:
        // the backend also returns 200 for content it silently skipped
        // (identical bytes already indexed under this or another name). Only
        // "added" or "replaced the previous version" actually land in the
        // index, so only those should grow the selection / the CTA's count.
        const indexed = uploaded.status === "added" || uploaded.status === "replaced the previous version";
        if (indexed) {
          if (!selected.includes(uploaded.file)) setSelected([...selected, uploaded.file]);
        } else {
          setError(new Error(`"${file.name}" wasn't added: ${uploaded.status}.`));
        }
      } catch (exc) {
        setError(exc as Error);
        break;
      } finally {
        setBusy(null);
      }
    }
    await refresh();
    onIndexChanged();
  }

  async function remove(path: string) {
    setBusy(path);
    setError(null);
    try {
      await api.removeDocument(path);
      if (result?.file === path) setResult(null);
      setSelected(selected.filter((p) => p !== path));
      await refresh();
      onIndexChanged();
    } catch (exc) {
      setError(exc as Error);
    } finally {
      setBusy(null);
    }
  }

  const limits = data?.limits;
  const full = !!data && !!limits && data.documents.length >= limits.max_documents;
  const uploading = busy !== null && !data?.documents.some((d) => d.path === busy);
  // Nothing happens until something is actually picked: a tick, or an upload
  // (which ticks itself). The seed sample sitting there untouched never
  // counts on its own, so there is no implicit "ask about all of them".
  const engaged = selected.length > 0;

  return (
    <div className="grid gap-4 sm:gap-6 lg:grid-cols-[20rem_minmax(0,1fr)]">
      {/* Left: bring your own. */}
      <div className="min-w-0 space-y-4">
        <Card title="Add a document" center reveal={null}>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); void send(e.dataTransfer.files); }}
            onClick={() => fileInput.current?.click()}
            className={`flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed p-6 text-center transition-colors sm:p-8 ${
              dragging
                ? "border-lime bg-lime/15"
                : "border-line bg-panel hover:border-lime hover:bg-lime/5"}`}>
            <input ref={fileInput} type="file" multiple hidden
                   accept={limits?.allowed_types.join(",")}
                   onChange={(e) => { void send(e.target.files); e.target.value = ""; }} />
            {uploading ? (
              <p className="text-base font-bold text-ink">Indexing {busy}...</p>
            ) : (
              <>
                <span className="grid size-12 place-items-center rounded-full bg-lime/15 text-lime">
                  <svg viewBox="0 0 24 24" fill="none" className="size-6">
                    <path d="M12 15.5V4M12 4 8 8M12 4l4 4" stroke="currentColor" strokeWidth="2"
                          strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M4.5 15.5v2.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-2.5"
                          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </span>
                <p className="text-base font-bold text-ink">Drop a file, or click</p>
                <p className="text-xs text-ink-soft">
                  {limits
                    ? `${limits.allowed_types.join(" ")} · up to ${limits.max_upload_mb} MB`
                    : "loading limits..."}
                </p>
              </>
            )}
          </div>

          <p className="mt-3 text-center text-sm text-ink-soft lg:mt-3 lg:text-left">
            <span className="font-bold text-ink">Not kept. </span>
            Deleted when you close this tab.
          </p>

          {full && (
            <p className="mt-2 text-center text-sm font-semibold text-ember lg:text-left">
              Limit is {limits?.max_documents} documents. Delete one first.
            </p>
          )}
          {error && <div className="mt-4"><ErrorBox error={error} /></div>}

          <div className="text-center lg:text-left">
            <Disclosure summary="What happens to it">
              Parsed, split into overlapping chunks on a token budget, embedded in
              one batched call and written into a versioned index. The file is
              hashed first, so uploading the same document twice costs nothing the
              second time.
            </Disclosure>
          </div>
        </Card>

        {result && (
          <Card title={result.file} tone="soft" reveal={null}>
            <div className="flex flex-wrap items-center gap-2">
              <Pill tone={result.status === "added" ? "good" : "warn"}>{result.status}</Pill>
              <Pill>{result.chunks} chunks</Pill>
              <Pill>{result.tokens.toLocaleString()} tokens</Pill>
              <Pill>{result.ingest_ms.toFixed(0)} ms</Pill>
            </div>
            {result.sample_chunks.length > 0 && (
              <Disclosure summary={`See the first ${result.sample_chunks.length} chunks`}>
                <div className="not-prose space-y-2">
                  {result.sample_chunks.map((chunk) => (
                    <div key={chunk.chunk_id} className="cue rounded-xl p-3">
                      <div className="mb-1 font-mono text-xs text-ink-soft">
                        {chunk.chunk_id}
                        {chunk.page !== null && ` · page ${chunk.page}`}
                        {` · ${chunk.tokens} tokens`}
                      </div>
                      <p className="text-xs leading-relaxed text-ink">{chunk.text}</p>
                    </div>
                  ))}
                </div>
              </Disclosure>
            )}
          </Card>
        )}
      </div>

      {/* Right: what is already here, and what to search. Always shown in
       *  full: with one seed document plus whatever a visitor adds, a
       *  collapse-behind-a-toggle treatment only hides the thing they need
       *  to reach (the CTA) and buys nothing, since the list is never long. */}
      <Card title="Tick what to search" center reveal={null}>
        {!data && !error && <Loading what="documents" />}

        <div className="space-y-2">
          {data?.documents.map((doc) => {
            const on = selected.includes(doc.path);
            return (
              <label key={doc.path}
                     className={`block cursor-pointer rounded-2xl border p-3 transition-colors sm:p-4 ${
                       on ? "border-lime bg-panel" : "border-line bg-card hover:border-ink-soft"}`}>
                <div className="flex min-w-0 items-start gap-3">
                  <input type="checkbox" checked={on} className="mt-1 size-4 shrink-0 accent-lime"
                         onChange={() => setSelected(on
                           ? selected.filter((p) => p !== doc.path)
                           : [...selected, doc.path])} />
                  <div className="min-w-0 flex-1">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="min-w-0 truncate text-base font-bold text-ink">{doc.path}</span>
                      {doc.uploaded && <Pill tone="lime">yours</Pill>}
                    </div>
                    {/* One badge instead of four differently coloured pills:
                     *  the old colours carried no meaning, just noise. Bold,
                     *  on the same violet as the chat bubble and evaluation
                     *  cards, so it reads as "the numbers that matter here"
                     *  rather than fine print a reader skims past. */}
                    <p className="mt-1.5 inline-flex flex-wrap items-center gap-x-1.5 rounded-full bg-violet px-3 py-1 text-xs font-bold text-on-violet">
                      <span className="tabular-nums">{doc.chunks} chunks</span>
                      <span aria-hidden className="opacity-60">·</span>
                      <span className="tabular-nums">{doc.tokens.toLocaleString()} tokens</span>
                      {doc.pages > 0 && (
                        <>
                          <span aria-hidden className="opacity-60">·</span>
                          <span className="tabular-nums">{doc.pages} pages</span>
                        </>
                      )}
                      <span aria-hidden className="opacity-60">·</span>
                      <span className="tabular-nums">{size(doc.size_bytes)}</span>
                    </p>
                  </div>
                </div>
                {/* stopPropagation, not preventDefault: this row sits inside the
                 *  <label>, and preventDefault on a bubbled click cancels the
                 *  *original* target's default action too, which silently killed
                 *  the View/Download links' own navigation. stopPropagation only
                 *  keeps the click from reaching the label (which would otherwise
                 *  toggle the checkbox), and leaves the links free to act. */}
                <div className="mt-3 flex items-center gap-2 pl-7"
                     onClick={(e) => e.stopPropagation()}>
                  <Ghost href={api.viewUrl(doc.path)} newTab>View</Ghost>
                  <Ghost href={api.downloadUrl(doc.path)} download>Download</Ghost>
                  {doc.uploaded && (
                    <Ghost onClick={() => void remove(doc.path)} disabled={busy === doc.path}>
                      {busy === doc.path ? "..." : "Delete"}
                    </Ghost>
                  )}
                </div>
              </label>
            );
          })}
        </div>

        <div className="mt-5 flex flex-wrap items-center justify-center gap-3 lg:justify-start">
          <Cta onClick={goToAsk} pulse={engaged} disabled={!engaged}>
            {engaged
              ? `Ask about ${selected.length} document${selected.length > 1 ? "s" : ""}`
              : "Select a document or upload yours"}
          </Cta>
          {selected.length > 0 && <Ghost onClick={() => setSelected([])}>Clear ticks</Ghost>}
        </div>
      </Card>
    </div>
  );
}
