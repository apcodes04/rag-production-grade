"""Make sure the container has an index before the API starts serving.

A fresh Hugging Face Space has an empty disk: no index directory, because the
index is a build artifact and build artifacts do not belong in git. Without
this, the first visitor sees an empty corpus and the demo looks broken before
they have done anything.

Ingesting the seed documents takes one embedding call for a handful of chunks,
so it costs about a second at boot and nothing afterwards. It is also
idempotent: on a restart where the disk survived, every file hashes the same
and nothing is re-embedded.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.factory import build_embedder, load_index  # noqa: E402
from rag.ingest import Ingestor  # noqa: E402
from rag.logging_setup import setup_logging  # noqa: E402
from rag.settings import Settings  # noqa: E402


def main() -> int:
    settings = Settings()
    setup_logging(settings.log_level)
    embedder = build_embedder(settings)
    index = load_index(settings, embedder)

    if index.chunks:
        print(f"index already has {len(index.chunks)} chunks, nothing to do")
        return 0

    if not settings.raw_dir.exists() or not any(settings.raw_dir.iterdir()):
        # Not an error: an empty demo that visitors fill themselves is valid.
        print(f"no seed documents in {settings.raw_dir}, starting with an empty index")
        return 0

    report = Ingestor(index, embedder, settings).ingest_dir()
    print(f"seeded index: {report.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
