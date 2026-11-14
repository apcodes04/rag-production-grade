export const PAGES = [
  { id: "documents", label: "Documents", blurb: "Choose or upload" },
  { id: "ask", label: "Chat", blurb: "Ask, and watch it work" },
  { id: "evaluation", label: "Evaluation", blurb: "Measured quality" },
] as const;

export type PageId = (typeof PAGES)[number]["id"];

export const isPage = (value: string): value is PageId => PAGES.some((p) => p.id === value);

/** The section switcher. Horizontal by default (three cards, or scrolling
 *  chips on a narrow screen). `vertical` stacks it as a narrow rail instead,
 *  for pairing beside a block that already spans the width below it. */
export function NavChips({ page, go, vertical = false }: {
  page: PageId; go: (next: PageId) => void; vertical?: boolean;
}) {
  return (
    <nav aria-label="Sections"
         className={vertical
           ? "flex w-20 shrink-0 flex-col gap-1.5"
           : "flex gap-2 overflow-x-auto pb-1 sm:grid sm:grid-cols-3 sm:gap-3 sm:overflow-visible sm:pb-0"}>
      {PAGES.map((item, i) => {
        const active = page === item.id;
        return (
          <button key={item.id} onClick={() => go(item.id)}
                  aria-current={active ? "page" : undefined}
                  className={vertical
                    ? `spot-card flex-1 rounded-xl border px-1.5 py-2 text-center ${
                        active ? "border-lime bg-panel" : "border-line bg-card"}`
                    : `spot-card topic-card shrink-0 rounded-2xl border px-4 py-2.5 text-left sm:shrink sm:p-4 ${
                        active ? "border-lime bg-panel" : "border-line bg-card"}`}>
            {vertical
              ? <span className="block text-xs font-bold leading-tight text-ink">{item.label}</span>
              : (
                <div className="flex items-baseline gap-2 sm:block">
                  <span className="text-xs font-bold uppercase tracking-wide text-ink-soft">
                    {i + 1}
                  </span>
                  <span className="text-base font-bold tracking-tight text-ink sm:mt-0.5 sm:block sm:text-lg">
                    {item.label}
                  </span>
                </div>
              )}
            {!vertical && <div className="hidden text-sm text-ink-soft sm:block">{item.blurb}</div>}
          </button>
        );
      })}
    </nav>
  );
}
