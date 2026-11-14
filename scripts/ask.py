"""Ask questions.  python scripts/ask.py "your question"   or no argument for a chat loop."""
import argparse
import sys

import _bootstrap  # noqa: F401

from rag.errors import RagError, RetryableProviderError, StoreError
from rag.factory import build_pipeline
from rag.logging_setup import setup_logging
from rag.settings import get_settings
from rag.types import Turn

HELP = "commands: /exit  /clear  /sources  /stats  /nocache <question>"


def show(answer, verbose: bool) -> None:
    print(f"\n{answer.text}\n")
    if answer.sources and (verbose or not answer.refused):
        for number, candidate in enumerate(answer.sources, start=1):
            chunk = candidate.chunk
            page = f" p{chunk.page}" if chunk.page else ""
            print(f"  [{number}] {chunk.source}{page}  "
                  f"dense={_fmt(candidate.dense_score)} sparse={_fmt(candidate.sparse_score)} "
                  f"rerank={_fmt(candidate.rerank_score)}")
    if answer.degraded:
        print(f"  WARNING: answered in degraded mode ({answer.degraded_reason}), "
              f"quality is lower than usual")
    print(f"  ({'cache hit' if answer.cache_hit else 'computed'}, "
          f"{sum(answer.timings_ms.values()):.0f} ms total, "
          f"grounded={_fmt(answer.groundedness)}, stages={answer.timings_ms})")


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


def safe_answer(pipeline, question, history=None, use_cache=True):
    """Turn library exceptions into one readable line. A stack trace is a bug
    report for us, not an answer for the person who asked a question."""
    try:
        return pipeline.answer(question, history or [], use_cache=use_cache)
    except RetryableProviderError as exc:
        print(f"\nThe model provider is unavailable right now (it was retried). "
              f"Try again in a moment, or set LLM_FALLBACK_MODEL in .env.\n  {exc}")
    except StoreError as exc:
        print(f"\nThe index cannot be used: {exc}\nRun: python scripts/ingest.py")
    except RagError as exc:
        print(f"\nRequest failed: {exc}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("WARNING" if not args.verbose else settings.log_level)
    pipeline = build_pipeline(settings)
    if not pipeline.index.chunks:
        print("Index is empty. Run: python scripts/ingest.py")
        sys.exit(1)

    if args.question:
        answer = safe_answer(pipeline, " ".join(args.question))
        if answer:
            show(answer, args.verbose)
        return

    print(f"Ask away. {HELP}")
    history: list[Turn] = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/clear":
            history.clear()
            print("history cleared")
            continue
        if question == "/stats":
            print(pipeline.index.stats())
            print(pipeline.cache.stats() if pipeline.cache else "cache disabled")
            continue
        use_cache = True
        if question.startswith("/nocache "):
            question, use_cache = question[9:], False

        answer = safe_answer(pipeline, question, history, use_cache)
        if not answer:
            continue
        show(answer, args.verbose)
        history.append(Turn("user", question))
        history.append(Turn("assistant", answer.text))


if __name__ == "__main__":
    main()
