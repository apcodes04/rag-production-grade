import { Suspense, lazy, useCallback, useEffect, useState } from "react";

import { api } from "./api";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Guide } from "./components/Guide";
import { isPage, NavChips, type PageId } from "./components/Nav";
import { PageMotion, useSpotlight } from "./components/PageMotion";
import { SmoothScroll } from "./components/SmoothScroll";
import { ErrorBox, Loading } from "./components/ui";
import { AskPage, type Exchange } from "./pages/Ask";
import { DocumentsPage } from "./pages/Documents";
import type { IndexStats } from "./types";

// Recharts is most of the bundle and only this page uses it.
const EvaluationPage = lazy(() =>
  import("./pages/Evaluation").then((m) => ({ default: m.EvaluationPage })));

/** Hash routing rather than a router dependency: three pages do not need one,
 *  and a hash URL survives a refresh on static hosting with no rewrite rules. */
function useHashRoute(): [PageId, (next: PageId) => void] {
  const read = (): PageId => {
    const raw = window.location.hash.replace("#/", "");
    return isPage(raw) ? raw : "documents";
  };
  const [page, setPage] = useState<PageId>(read);

  useEffect(() => {
    const onChange = () => {
      setPage(read());
      window.scrollTo({ top: 0, behavior: "instant" });
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return [page, (next) => { window.location.hash = `#/${next}`; }];
}

export default function App() {
  const [page, go] = useHashRoute();
  const [health, setHealth] = useState<IndexStats | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [thread, setThread] = useState<Exchange[]>([]);
  const [selected, setSelected] = useState<string[]>([]);

  useSpotlight();

  const refreshHealth = useCallback(() => {
    void api.health().then(setHealth).catch((exc) => setError(exc as Error));
  }, []);

  useEffect(refreshHealth, [refreshHealth]);

  // A visitor's uploads are deleted when they leave. pagehide fires in cases
  // where unload does not, notably when a mobile browser freezes the tab.
  useEffect(() => {
    const onLeave = () => api.cleanupUploadsOnLeave();
    window.addEventListener("pagehide", onLeave);
    return () => window.removeEventListener("pagehide", onLeave);
  }, []);

  const last = thread.length ? thread[thread.length - 1].answer : null;

  return (
    <>
      <SmoothScroll />
      <PageMotion page={page} />

      <div className={`mx-auto max-w-6xl px-4 sm:px-6 sm:pb-24 sm:pt-10 ${
        page === "ask" ? "pb-3 pt-3" : "pb-24 pt-8"}`}>
        {/* Mobile, Documents and Evaluation pages: the title and blurb take
         *  the left block, the section rail stands beside them on the right,
         *  both visible together at the top — the same split used on the
         *  Chat page for "What happens inside" and its rail. Desktop is
         *  untouched: the header spans full width, the strip sits below it. */}
        {(page === "documents" || page === "evaluation") && (
          <div className="flex items-start gap-3 lg:hidden">
            <div className="min-w-0 flex-1">
              <h1 className="text-xl font-extrabold leading-[1.05] tracking-tighter text-ink">
                Production grade RAG
              </h1>
              <p className="mt-1.5 text-sm text-ink-soft">
                Hybrid retrieval, dense vectors fused with a hand written BM25.
                Guardrails on the input, the retrieval, the prompt and the output.
                No LangChain, no LlamaIndex.
              </p>
            </div>
            <NavChips page={page} go={go} vertical />
          </div>
        )}

        <header className={`${
          page === "ask" ? "hidden sm:block"
            : page === "documents" || page === "evaluation" ? "hidden lg:block" : ""
        } mb-6 sm:mb-8`}>
          <h1 className="text-2xl font-extrabold leading-[1.05] tracking-tighter text-ink sm:text-5xl">
            Production grade RAG
          </h1>
          <p className="mt-2 max-w-3xl text-base text-ink-soft sm:mt-3 sm:text-lg">
            Hybrid retrieval, dense vectors fused with a hand written BM25.
            Guardrails on the input, the retrieval, the prompt and the output.
            <span className="hidden sm:inline">
              {" "}Incremental ingestion into a versioned index, an answer cache,
              retries with a deadline, a fallback model, and an evaluation harness
              that measures all of it.
            </span>{" "}
            No LangChain, no LlamaIndex.
          </p>
        </header>

        {/* On the Chat and Documents pages, the rail moves beside the left
         *  block on narrow screens instead (above), so this horizontal
         *  strip is only for the other pages, or wide screens, there. */}
        <div className={page === "ask" || page === "documents" || page === "evaluation" ? "hidden lg:block" : ""}>
          <div className={page === "ask" ? "mb-3 pt-1" : "mb-6"}>
            <NavChips page={page} go={go} />
          </div>
        </div>

        {error && <div className="mb-6"><ErrorBox error={error} /></div>}

        <main id="main" className="space-y-6">
          <ErrorBoundary>
            {page === "documents" && (
              <DocumentsPage onIndexChanged={refreshHealth} goToAsk={() => go("ask")}
                             selected={selected} setSelected={setSelected} />
            )}
            {page === "ask" && (
              <AskPage thread={thread} setThread={setThread} health={health}
                       documentCount={health?.documents ?? null}
                       selected={selected} goToDocuments={() => go("documents")}
                       page={page} go={go} />
            )}
            {page === "evaluation" && (
              <Suspense fallback={<Loading what="the evaluation charts" />}>
                <EvaluationPage />
              </Suspense>
            )}
          </ErrorBoundary>
        </main>
      </div>

      <Guide page={page}
             documentCount={health?.documents ?? 0}
             selectedCount={selected.length}
             askedCount={thread.length}
             lastRefused={last?.refused ?? false} />
    </>
  );
}
