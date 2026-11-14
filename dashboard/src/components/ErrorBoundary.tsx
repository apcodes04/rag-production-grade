import { Component, type ErrorInfo, type ReactNode } from "react";

/** Catches a render crash and shows it, instead of unmounting the whole app.
 *
 *  Without this, one undefined field in one row blanks the entire page to
 *  white, which is both the worst possible thing to happen during a demo and
 *  the least informative: no message, no stack, nothing to act on.
 */
export class ErrorBoundary extends Component<
  { children: ReactNode }, { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("render failed", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="rounded-3xl border border-bad bg-bad-bg p-6">
        <h2 className="text-xl font-bold text-bad">This section failed to render.</h2>
        <p className="mt-2 text-base text-ink-soft">
          The rest of the page still works. The error was:
        </p>
        <pre className="mt-3 overflow-x-auto rounded-2xl bg-panel p-4 font-mono text-sm text-ink">
          {this.state.error.message}
        </pre>
        <button onClick={() => this.setState({ error: null })}
                className="mt-4 rounded-full border-2 border-line px-4 py-1.5 text-sm font-semibold text-ink">
          Try again
        </button>
      </div>
    );
  }
}
