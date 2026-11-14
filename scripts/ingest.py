"""Build or update the index.  python scripts/ingest.py [--dir data/raw] [--prune]"""
import argparse
import json
import logging

import _bootstrap  # noqa: F401

from rag.factory import build_embedder, load_index
from rag.ingest import Ingestor
from rag.logging_setup import new_request_id, setup_logging
from rag.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the index")
    parser.add_argument("paths", nargs="*", help="specific files (default: everything in raw_dir)")
    parser.add_argument("--prune", action="store_true",
                        help="remove documents that no longer exist on disk")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    new_request_id()
    logging.getLogger("ingest").info("starting", extra={"provider": settings.provider})

    embedder = build_embedder(settings)
    index = load_index(settings, embedder)
    ingestor = Ingestor(index, embedder, settings)
    report = (ingestor.ingest([__import__("pathlib").Path(p) for p in args.paths],
                              remove_missing=args.prune)
              if args.paths else ingestor.ingest_dir(remove_missing=args.prune))
    print(json.dumps(report.summary(), indent=2))
    print(json.dumps(index.stats(), indent=2))
    if report.failed:
        print("FAILED:", report.failed)


if __name__ == "__main__":
    main()
