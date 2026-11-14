"""Measure the pipeline against the golden set.

  python scripts/evaluate.py
  python scripts/evaluate.py --set eval/golden_set.json --judge --tag chunk512

Any change to chunking, embeddings, fusion, reranking or prompts must be
justified by a before and after run of this script. Reports are saved to
eval/reports/ so the comparison survives the conversation.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from rag.evaluation.harness import evaluate, format_report, load_golden
from rag.factory import build_pipeline
from rag.logging_setup import setup_logging
from rag.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="eval/golden_set.json")
    parser.add_argument("--judge", action="store_true", help="also run the LLM judge (costs tokens)")
    parser.add_argument("--tag", default="run", help="name for the saved report")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel questions (the API call is blocking IO)")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("WARNING")
    pipeline = build_pipeline(settings)
    report = evaluate(pipeline, load_golden(Path(args.set)), use_judge=args.judge,
                      workers=args.workers)
    print(format_report(report))

    out_dir = Path("eval/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["created_at"].replace(":", "").replace("-", "")
    path = out_dir / f"{args.tag}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
