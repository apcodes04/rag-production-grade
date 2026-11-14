"""The index: chunks, their vectors and their BM25 postings, saved as ONE unit.

Two design decisions worth defending in an interview:

1. One bundle, not three stores. The reference project saved FAISS and BM25
   separately, and its two add paths disagreed (one appended, one replaced),
   so the dense and sparse views of the corpus could silently drift apart.
   Here chunks are the single source of truth and both indexes are derived
   from them, saved together or not at all.

2. Versioned directories plus a tiny pointer file, instead of overwriting.
   Writing v3/ then atomically replacing current.json means a crash halfway
   through a reindex leaves the old index fully intact, and a rollback is one
   line in a text file. This is the blue/green deploy pattern applied to data.

File formats and why:
  manifest.json  small, human readable, the first thing you open when
                 debugging; holds the contract (model, dim, versions).
  chunks.jsonl   one JSON object per line: appendable, streamable, greppable,
                 diffable in git. A single huge JSON array would need to be
                 parsed whole just to read one chunk.
  vectors.npy    binary float32 matrix: exact round trip, loads with mmap,
                 about 3 KB per 768 dim vector versus about 15 KB as JSON text.
  bm25.jsonl     the tokens per chunk, so a load does not have to re tokenise
                 and the tokenizer version stays pinned to the index.
  NO pickle anywhere: unpickling a file someone else can write is remote code
  execution. The reference project pickled both of its stores.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rag.errors import StoreError
from rag.stores.bm25_index import BM25Index
from rag.stores.vector_index import VectorIndex
from rag.tokenizer import TOKENIZER_VERSION
from rag.types import Chunk

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CURRENT_POINTER = "current.json"


@dataclass
class DocRecord:
    """What we know about one ingested file, so we can skip it next time."""
    path: str
    doc_id: str
    sha256: str
    size: int
    mtime_ns: int
    chunk_ids: list[str] = field(default_factory=list)


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file in the same directory, then rename. A reader never
    sees a half written file, because rename is atomic on POSIX and Windows."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        write(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class Index:
    def __init__(self, dim: int, embedding_model: str, backend: str = "numpy"):
        self.dim = dim
        self.embedding_model = embedding_model
        self.chunks: dict[str, Chunk] = {}
        self.vectors = VectorIndex(dim, backend)
        self.bm25 = BM25Index()
        self.docs: dict[str, DocRecord] = {}

    # ------------------------------------------------------------- mutation
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise StoreError("chunks and vectors length mismatch")
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.bm25.add(chunk.chunk_id, chunk.text)
        self.vectors.add([c.chunk_id for c in chunks], vectors)
        self.bm25.build()

    def remove_doc(self, path: str) -> int:
        record = self.docs.pop(path, None)
        if not record:
            return 0
        for chunk_id in record.chunk_ids:
            self.chunks.pop(chunk_id, None)
            self.bm25.remove(chunk_id)
        self.vectors.keep_only(set(self.chunks))
        self.bm25.build()
        return len(record.chunk_ids)

    # -------------------------------------------------------------- identity
    def index_version(self) -> str:
        """Changes whenever the content or the embedding contract changes.
        Used in cache keys so a reindex cannot serve stale answers."""
        digest = hashlib.sha256()
        digest.update(f"{self.embedding_model}|{self.dim}|{TOKENIZER_VERSION}|".encode())
        for chunk_id in sorted(self.chunks):
            digest.update(chunk_id.encode())
        return digest.hexdigest()[:12]

    def stats(self) -> dict:
        return {"chunks": len(self.chunks), "vectors": len(self.vectors),
                "bm25_docs": len(self.bm25), "documents": len(self.docs),
                "dim": self.dim, "embedding_model": self.embedding_model,
                "index_version": self.index_version()}

    # ----------------------------------------------------------- persistence
    def save(self, root: Path) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        existing = [int(p.name[1:]) for p in root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
        version_dir = root / f"v{max(existing, default=0) + 1}"
        version_dir.mkdir()

        ordered = [self.chunks[cid] for cid in self.vectors.ids]
        _atomic_write(version_dir / "chunks.jsonl", lambda h: h.write(
            "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in ordered).encode()))
        _atomic_write(version_dir / "vectors.npy", lambda h: np.save(h, self.vectors.vectors))
        _atomic_write(version_dir / "bm25.jsonl", lambda h: h.write(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.bm25.to_records()).encode()))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model": self.embedding_model, "dim": self.dim,
            "tokenizer_version": TOKENIZER_VERSION, "index_version": self.index_version(),
            "counts": {"chunks": len(self.chunks), "documents": len(self.docs)},
            "docs": {path: asdict(record) for path, record in self.docs.items()},
        }
        _atomic_write(version_dir / "manifest.json",
                      lambda h: h.write(json.dumps(manifest, indent=2).encode()))
        # Promote last: until this line runs, readers still see the old version.
        _atomic_write(root / CURRENT_POINTER, lambda h: h.write(
            json.dumps({"version": version_dir.name,
                        "written_at": manifest["created_at"]}, indent=2).encode()))
        log.info("index saved", extra={"version": version_dir.name, **self.stats()})
        return version_dir

    @classmethod
    def load(cls, root: Path, *, dim: int, embedding_model: str,
             backend: str = "numpy") -> Index:
        root = Path(root)
        pointer = root / CURRENT_POINTER
        index = cls(dim, embedding_model, backend)
        if not pointer.exists():
            return index                       # fresh start, not an error
        version_dir = root / json.loads(pointer.read_text())["version"]
        manifest = json.loads((version_dir / "manifest.json").read_text())
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise StoreError(f"index schema {manifest['schema_version']} != {SCHEMA_VERSION}, reindex")
        if manifest["embedding_model"] != embedding_model or manifest["dim"] != dim:
            raise StoreError(
                f"index was built with {manifest['embedding_model']} dim {manifest['dim']}, "
                f"settings say {embedding_model} dim {dim}. Reindex or fix the settings.")
        if manifest.get("tokenizer_version") != TOKENIZER_VERSION:
            raise StoreError("tokenizer changed since this index was built, reindex")

        chunks = [Chunk.from_dict(json.loads(line))
                  for line in (version_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        index.chunks = {c.chunk_id: c for c in chunks}
        with open(version_dir / "vectors.npy", "rb") as handle:
            vectors = np.load(handle)
        if len(vectors) != len(chunks):
            raise StoreError("vectors and chunks are out of sync, reindex")
        index.vectors.add([c.chunk_id for c in chunks], vectors)
        records = [json.loads(line) for line in
                   (version_dir / "bm25.jsonl").read_text(encoding="utf-8").splitlines() if line]
        index.bm25 = BM25Index.from_records(records)
        index.docs = {path: DocRecord(**record) for path, record in manifest["docs"].items()}
        log.info("index loaded", extra={"version": version_dir.name, **index.stats()})
        return index
