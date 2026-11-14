"""Which models can this key ACTUALLY call?  python scripts/probe_models.py

`list_models.py` asks the API what exists. That is not the same question.
A model can be listed and still answer 404 NOT_FOUND ("no longer available to
new users"), because legacy models are often frozen to the accounts that
already used them. The only honest test is to call it.

Use this to choose LLM_MODEL and LLM_FALLBACK_MODEL. An unverified fallback is
not a fallback: it fails exactly when the primary is already failing.
"""
import argparse
import time

import _bootstrap  # noqa: F401

from rag.settings import get_settings

DEFAULT_CANDIDATES = [
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=None,
                        help="models to probe (default: the common flash family)")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    from google import genai
    from google.genai import types

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value(),
                          http_options=types.HttpOptions(timeout=int(args.timeout * 1000)))
    candidates = args.models or DEFAULT_CANDIDATES
    print(f"{'model':28} {'result':10} {'ms':>6}  detail")
    for model in candidates:
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model, contents="Reply with the single word: ok",
                config=types.GenerateContentConfig(max_output_tokens=512))
            elapsed = (time.perf_counter() - started) * 1000
            text = (getattr(response, "text", None) or "").strip()[:20]
            meta = getattr(response, "usage_metadata", None)
            thinking = getattr(meta, "thoughts_token_count", 0) or 0
            print(f"{model:28} {'OK':10} {elapsed:6.0f}  {text!r} thinking={thinking}")
        except Exception as exc:                 # noqa: BLE001 probing is the point
            elapsed = (time.perf_counter() - started) * 1000
            message = str(exc).split("{")[0].strip()[:70]
            print(f"{model:28} {'FAILED':10} {elapsed:6.0f}  {message}")
    print("\nPick a primary and a fallback that both say OK, and prefer two that "
          "are unlikely to be busy at the same time (for example a flash and a flash-lite).")


if __name__ == "__main__":
    main()
