"""Turning files on disk into an index, incrementally.

The reference project had one ingestion mode: if an index file exists, load it
and ignore everything else. Drop a new PDF into the folder and it was silently
never indexed (audit bug B5). Re-run with force and it re-embedded everything,
which costs real money on every change.

This ingester remembers a sha256 per file in the manifest, so:
  unchanged file  -> skipped, zero API calls
  changed file    -> its old chunks are dropped and only it is re-embedded
  new file        -> embedded
  deleted file    -> its chunks leave the index (with remove_missing=True)
  same bytes under a new name -> reported as a duplicate and skipped
Everything is written as a new index version at the end, so a failure halfway
through leaves the previous index untouched.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunker import chunk_pages
from rag.loader import SUPPORTED_SUFFIXES, load_document
from rag.providers.base import EmbeddingProvider
from rag.settings import Settings
from rag.stores.index import DocRecord, Index

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_added: int = 0
    chunks_removed: int = 0
    duration_s: float = 0.0
    index_version: str = ""

    def summary(self) -> dict:
        return {"added": len(self.added), "updated": len(self.updated),
                "skipped": len(self.skipped), "removed": len(self.removed),
                "duplicates": len(self.duplicates), "failed": len(self.failed),
                "chunks_added": self.chunks_added, "chunks_removed": self.chunks_removed,
                "duration_s": round(self.duration_s, 2), "index_version": self.index_version}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Ingestor:
    def __init__(self, index: Index, embedder: EmbeddingProvider, settings: Settings):
        self.index = index
        self.embedder = embedder
        self.settings = settings

    def _key(self, path: Path) -> str:
        """Stable identity for a document: its path relative to whichever
        document root holds it.

        Two roots exist (the committed seed corpus and the upload folder), and
        a file must key the same way whichever one it sits in, because that key
        is what makes re-ingesting the same file a no-op instead of a duplicate.
        """
        resolved = path.resolve()
        for root in self.settings.document_roots():
            try:
                return resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
        return resolved.as_posix()

    def ingest(self, paths: list[Path], *, remove_missing: bool = False,
               save: bool = True) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport()
        paths = [Path(p) for p in paths]
        seen_keys = {self._key(path) for path in paths}

        # Prune BEFORE indexing, never after. A rename is a delete plus an add,
        # and the two must be applied in that order: pruning last meant the new
        # name was first seen as a duplicate of the old entry and skipped, then
        # the old entry was removed as missing, so a renamed file disappeared
        # from the index completely.
        if remove_missing:
            for key in [k for k in self.index.docs if k not in seen_keys]:
                report.chunks_removed += self.index.remove_doc(key)
                report.removed.append(key)

        for path in paths:
            key = self._key(path)
            if not path.exists():
                report.failed.append((key, "file not found"))
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                report.failed.append((key, f"unsupported type {path.suffix}"))
                continue

            stat = path.stat()
            digest = file_sha256(path)
            known = self.index.docs.get(key)
            if known and known.sha256 == digest:
                report.skipped.append(key)
                continue
            duplicate = next((other for other, record in self.index.docs.items()
                              if record.sha256 == digest and other != key), None)
            if duplicate:
                log.warning("duplicate content, skipping",
                            extra={"path": key, "same_as": duplicate})
                report.duplicates.append(key)
                continue

            if known:                       # changed file: drop the old chunks first
                report.chunks_removed += self.index.remove_doc(key)
                report.updated.append(key)
            else:
                report.added.append(key)

            try:
                pages = load_document(path)
                chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
            except Exception as exc:        # noqa: BLE001 one bad file must not kill the run
                log.error("failed to read document", extra={"path": key, "error": str(exc)[:200]})
                report.failed.append((key, str(exc)[:200]))
                for bucket in (report.added, report.updated):
                    if key in bucket:
                        bucket.remove(key)
                continue

            if chunks:
                vectors = self.embedder.embed_documents([c.text for c in chunks])
                self.index.add(chunks, vectors)
                report.chunks_added += len(chunks)
            else:
                log.warning("no text extracted, nothing indexed", extra={"path": key})

            self.index.docs[key] = DocRecord(
                path=key, doc_id=(chunks[0].doc_id if chunks else digest[:12]), sha256=digest,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                chunk_ids=[c.chunk_id for c in chunks])

        # Only write a new index version when the index actually changed. A
        # no-op run used to copy the whole vector file to a new version folder,
        # which is free on 6 chunks and expensive on a million.
        changed = bool(report.added or report.updated or report.removed)
        if save and changed:
            self.index.save(self.settings.index_dir)
        elif save:
            log.info("nothing changed, index not rewritten")
        report.index_version = self.index.index_version()
        report.duration_s = time.perf_counter() - started
        log.info("ingest finished", extra=report.summary())
        return report

    def ingest_dir(self, directory: Path | None = None, **kwargs) -> IngestReport:
        """Ingest a directory, or every document root when none is named.

        Scanning both roots matters for remove_missing: given only the seed
        corpus, a prune would delete every uploaded document as "missing".
        """
        roots = [Path(directory)] if directory else self.settings.document_roots()
        files = sorted(path for root in roots if root.exists()
                       for path in root.rglob("*")
                       if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
        return self.ingest(files, **kwargs)
