import { Card } from "./ui";

/** Numbered instructions, deliberately separate from the live readouts: this
 *  panel says what to DO, the panels below report what the system DID. */
export function HowToRead() {
  const steps = [
    ["Step 1", "Pick an evaluation run in the comparison panel below. Each run is one configuration measured against the same 21 question golden set."],
    ["Step 2", "Read the retrieval metrics first. The generator can only use what retrieval found, so recall@k caps everything downstream."],
    ["Step 3", "Check the two refusal numbers together. Correct refusals should be 1.0 and false refusals 0.0. One without the other means nothing."],
    ["Step 4", "Open the calibration chart. If the in scope and out of scope score clouds overlap, no similarity threshold can separate them, and the prompt is doing the work."],
    ["Step 5", "Use the question table to drill into any single row: the answer, the chunks that were retrieved, and every score they collected."],
    ["Step 6", "Switch to the Ask page to run the same pipeline live on your own documents, then to Pipeline to see per stage timings for that question."],
  ];
  return (
    <Card title="How to read these numbers" hint="Each number below is measured, never estimated.">
      <ol className="grid gap-3 md:grid-cols-2">
        {steps.map(([step, text]) => (
          <li key={step} className="flex gap-3 text-sm leading-relaxed">
            <span className="shrink-0 rounded bg-lime px-2 py-0.5 font-mono text-xs text-sel">
              {step}
            </span>
            <span className="text-ink">{text}</span>
          </li>
        ))}
      </ol>
    </Card>
  );
}
