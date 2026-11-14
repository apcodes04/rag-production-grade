import { useEffect, useState } from "react";

export interface GuideState {
  page: string;
  documentCount: number;
  selectedCount: number;
  askedCount: number;
  lastRefused: boolean;
}

/** The one next step, worked out from where the visitor actually is.
 *
 *  This exists so the pages do not have to explain themselves. Every sentence
 *  of guidance that used to sit on a page now lives here, shown one at a time,
 *  which is the difference between a product and a manual.
 */
function nextStep(s: GuideState): { title: string; body: string; cta?: string } {
  if (s.page === "documents") {
    if (s.documentCount === 0) {
      return { title: "Add a document", body: "Drop in a PDF, Word file, CSV or text file. It is chunked, embedded and searchable in about a second." };
    }
    if (s.selectedCount === 0) {
      return {
        title: "Pick what to search",
        body: "Tick one document to see retrieval scoped to it, or leave everything unticked to search the whole corpus. Your own upload is deleted when you close this tab.",
        cta: "Then press Ask",
      };
    }
    return { title: "Ready", body: `${s.selectedCount} document${s.selectedCount > 1 ? "s" : ""} selected. Answers can only come from what you picked.`, cta: "Press Ask" };
  }

  if (s.page === "ask") {
    if (s.askedCount === 0) {
      return {
        title: "Ask anything",
        body: "Watch the panel on the right as the answer arrives: it shows how long each of the seven stages took. Then try one of the red buttons, which are meant to be refused.",
      };
    }
    if (s.lastRefused) {
      return {
        title: "That refusal was the point",
        body: "A system that answers everything is a system that invents things. Open 'How did it come up with this answer' to see which guardrail stopped it.",
      };
    }
    return {
      title: "Check the evidence",
      body: "Open 'How did it come up with this answer' under the reply. Every cited chunk is shown with its dense and keyword scores, so you can verify the answer against its source.",
    };
  }

  return {
    title: "These numbers are recorded",
    body: "Measured runs against a labelled question set, replayable from JSON in the repo. They do not change when you upload a document, because nobody has labelled the right answers for it.",
  };
}

export function Guide(props: GuideState) {
  const [open, setOpen] = useState(false);
  const step = nextStep(props);

  // Moving to a different part of the app means the advice changed, so the
  // badge goes back to unread.
  const [seen, setSeen] = useState("");
  const id = `${props.page}:${step.title}`;
  useEffect(() => { setOpen(false); }, [props.page]);

  return (
    <div className={`fixed right-5 z-40 flex flex-col items-end gap-3 ${
      // The mobile composer on the Ask page is a fixed bar pinned to the
      // bottom of the screen, under this button's old spot. Clear it there;
      // every other page keeps the button tucked in the corner.
      props.page === "ask" ? "bottom-36 lg:bottom-5" : "bottom-5"}`}>
      {open && (
        <div className="menu-pop w-[min(22rem,calc(100vw-2.5rem))] rounded-3xl border border-line bg-card p-5 shadow-2xl">
          <h3 className="text-base font-bold text-ink">{step.title}</h3>
          <p className="mt-1.5 text-sm leading-relaxed text-ink-soft">{step.body}</p>
          {step.cta && (
            <p className="mt-3 text-sm font-bold text-ink">{step.cta}</p>
          )}
        </div>
      )}

      <button type="button"
              aria-label={open ? "Hide the guide" : "What should I do here?"}
              aria-expanded={open}
              onClick={() => { setOpen(!open); setSeen(id); }}
              className="cta relative grid size-14 place-items-center rounded-full bg-lime text-2xl text-on-lime">
        {open ? "×" : "?"}
        {!open && seen !== id && (
          <span aria-hidden
                className="absolute -right-0.5 -top-0.5 size-3.5 rounded-full border-2 border-paper bg-ember" />
        )}
      </button>
    </div>
  );
}
