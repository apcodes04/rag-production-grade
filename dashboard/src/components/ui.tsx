import type { ReactNode } from "react";

type Tone = "plain" | "deep" | "slate" | "soft" | "violet";

const CARD_TONES: Record<Tone, string> = {
  plain: "border-line bg-card surface",
  // Deep purple, borrowed from the dark palette. For the one section on a
  // page that should read as the evidence behind everything else.
  deep: "border-deep bg-deep text-on-deep",
  slate: "border-slate bg-slate text-on-slate",
  soft: "border-line bg-panel surface",
  // The same purple as the chat bubble, so the evaluation numbers read as
  // "the same voice talking back to you" rather than a different section.
  violet: "border-violet bg-violet text-on-violet",
};

export function Card({ title, hint, children, className = "", reveal = "up", tone = "plain", center = false }: {
  title?: string; hint?: ReactNode; children: ReactNode;
  className?: string; reveal?: string | null; tone?: Tone; center?: boolean;
}) {
  const dark = tone === "deep" || tone === "slate" || tone === "violet";
  const hintColor = tone === "violet" ? "text-on-violet/80" : dark ? "text-deep-soft" : "text-ink-soft";
  return (
    <section {...(reveal ? { "data-rv": reveal } : {})}
             className={`min-w-0 rounded-3xl border p-6 sm:p-7 ${CARD_TONES[tone]} ${className}`}>
      {title && (
        <header className={`mb-5 ${center ? "text-center lg:text-left" : ""}`}>
          <h2 className={`text-xl font-bold tracking-tight sm:text-2xl ${dark ? "text-on-deep" : "text-ink"}`}>
            {title}
          </h2>
          {hint && (
            <p className={`mt-1.5 text-base leading-relaxed ${hintColor}`}>
              {hint}
            </p>
          )}
        </header>
      )}
      {children}
    </section>
  );
}

/** Detail that an interviewer can open if they want it, and ignore if they do not.
 *
 *  The depth is the point of this project, but a wall of explanation on first
 *  load reads as noise. Collapsed by default keeps both: a clean page, and the
 *  reasoning one click away.
 */
export function Disclosure({ summary, children, tone = "light" }: {
  summary: string; children: ReactNode; tone?: "light" | "dark";
}) {
  const summaryCls = tone === "dark"
    ? "border-white/20 text-coal-soft hover:border-lime hover:text-on-coal"
    : "border-line text-ink-soft hover:border-lime hover:text-ink";
  const bodyCls = tone === "dark" ? "text-coal-soft" : "text-ink-soft";
  return (
    <details className="group mt-4">
      <summary className={`inline-flex cursor-pointer list-none items-center gap-2 rounded-full border px-4 py-1.5 text-sm font-semibold transition-colors ${summaryCls}`}>
        <span className="transition-transform group-open:rotate-90">›</span>
        {summary}
      </summary>
      <div className={`read mt-3 ${bodyCls}`}>{children}</div>
    </details>
  );
}

/** A number with the rule for reading it. A metric with no stated direction
 *  is decoration, so every stat carries its own sense of good. */
export function Stat({ label, value, suffix = "", good, explain }: {
  label: string; value: number | string; suffix?: string;
  good?: boolean | null; explain?: string;
}) {
  const tone = good === true ? "text-sel" : good === false ? "text-ember" : "text-ink";
  return (
    <div className="rounded-2xl border border-line bg-panel p-4">
      <div className="text-xs font-semibold tracking-wide text-ink-soft">{label}</div>
      <div className={`mt-1 text-3xl font-extrabold tabular-nums tracking-tight ${tone}`}>
        {value}
        <span className="ml-0.5 text-base font-bold text-ink-soft">{suffix}</span>
      </div>
      {explain && <div className="mt-1.5 text-sm leading-snug text-ink-soft">{explain}</div>}
    </div>
  );
}

const META_TONES = {
  chunks: "bg-sel/15 text-sel",
  tokens: "bg-brand/15 text-brand",
  pages: "bg-spot text-on-spot",
  size: "bg-line/40 text-ink-soft",
} as const;

export function Meta({ kind, children }: {
  kind: keyof typeof META_TONES; children: ReactNode;
}) {
  return (
    <span className={`rounded-md px-2 py-0.5 text-xs font-bold tabular-nums ${META_TONES[kind]}`}>
      {children}
    </span>
  );
}

export function Pill({ children, tone = "neutral" }: {
  children: ReactNode; tone?: "neutral" | "good" | "bad" | "warn" | "lime";
}) {
  const tones = {
    neutral: "border-line text-ink-soft",
    good: "border-sel text-sel",
    bad: "border-bad text-bad",
    warn: "border-spot text-ink",
    lime: "border-lime bg-lime text-on-lime",
  } as const;
  return (
    <span className={`rounded-full border px-3 py-1 text-sm font-semibold ${tones[tone]}`}>
      {children}
    </span>
  );
}

/** The one action that moves a visitor forward. Lime, raised, pressable.
 *  Only one of these per screen. */
export function Cta({ children, onClick, disabled, pulse = false, type = "button" }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean;
  pulse?: boolean; type?: "button" | "submit";
}) {
  return (
    <button type={type} onClick={onClick} disabled={disabled}
            className={`cta relative rounded-full bg-lime px-8 py-3.5 text-lg font-bold text-on-lime disabled:opacity-50 ${pulse ? "cta-pulse" : ""}`}>
      {children}
    </button>
  );
}

/** Quiet secondary action: outlined, never filled, never lime. */
export function Ghost({ children, onClick, disabled, href, download, newTab = false, danger = false }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean;
  href?: string; download?: boolean; newTab?: boolean; danger?: boolean;
}) {
  // `danger` marks the examples that are supposed to be refused, so a visitor
  // can see from the colour that failing is the expected outcome.
  const className = danger
    ? "inline-flex items-center rounded-full border-2 border-bad bg-bad-bg px-4 py-1.5 text-sm font-semibold text-bad transition-colors hover:border-bad disabled:opacity-40"
    : "inline-flex items-center rounded-full border-2 border-line px-4 py-1.5 text-sm font-semibold text-ink-soft transition-colors hover:border-ink hover:text-ink disabled:opacity-40";
  return href
    ? <a href={href} download={download} className={className}
         {...(newTab ? { target: "_blank", rel: "noopener noreferrer" } : {})}>{children}</a>
    : <button type="button" onClick={onClick} disabled={disabled} className={className}>{children}</button>;
}

export function Loading({ what }: { what: string }) {
  return <div className="animate-pulse text-base text-ink-soft">Loading {what}...</div>;
}

export function ErrorBox({ error }: { error: Error }) {
  return (
    <div className="rounded-2xl border border-bad bg-bad-bg p-4 text-base text-bad">
      <strong className="font-bold">That did not work. </strong>{error.message}
    </div>
  );
}

export const fmt = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined ? "-" : value.toFixed(digits);
export const pct = (value: number | null | undefined) =>
  value === null || value === undefined ? "-" : `${Math.round(value * 100)}%`;
