"""Load documents into clean, page-tagged text.

Why a separate loader: garbage in, garbage out. Most "bad RAG" is bad text
extraction (hyphen breaks, repeated headers, wrapped lines), not a bad model.
Fixes vs reference: stable doc_id from file bytes, repeated header/footer
removal, hyphen and line-wrap repair, explicit errors for unsupported types,
warning for empty (scanned) pages.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from rag.types import Page

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".csv"}


def file_doc_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


# ---------------------------------------------------------------- cleaning

_HYPHEN_BREAK = re.compile(r"(\w)-\n([a-z])")   # "exam-\nple" -> "example"
_SPACES = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")
_SENTENCE_END = (".", "!", "?", ":", ";")
_LIST_ITEM = re.compile(r"^([-*•]|\d+[.)])\s")


def normalize(text: str) -> str:
    """Unicode + whitespace cleanup. Keeps single newlines and blank lines."""
    text = unicodedata.normalize("NFKC", text)        # ligatures, full-width chars
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def unwrap_lines(text: str) -> str:
    """Join PDF line wraps inside a paragraph (heuristic, see audit note)."""
    paragraphs = []
    for para in text.split("\n\n"):
        lines = para.split("\n")
        merged = [lines[0]]
        for line in lines[1:]:
            prev = merged[-1]
            wrapped = (
                bool(prev) and bool(line)
                and not prev.endswith(_SENTENCE_END)
                and not _LIST_ITEM.match(line)
                and (line[0].islower() or len(prev) >= 60)
            )
            if wrapped:
                merged[-1] = prev + " " + line
            else:
                merged.append(line)
        paragraphs.append("\n".join(merged))
    return "\n\n".join(paragraphs)


def _signature(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())    # "Page 3" == "Page 4"


def _edge_indices(lines: list[str]) -> set[int]:
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    return set(nonempty[:2] + nonempty[-2:])


def strip_repeated_edges(texts: list[str], min_pages: int = 4, threshold: float = 0.6) -> list[str]:
    """Remove running headers/footers: short lines at page edges that repeat on most pages."""
    if len(texts) < min_pages:
        return texts
    counts: Counter[str] = Counter()
    for text in texts:
        lines = text.split("\n")
        edge = _edge_indices(lines)
        counts.update({_signature(lines[i]) for i in edge if len(lines[i]) <= 80})
    repeated = {sig for sig, n in counts.items() if n >= threshold * len(texts)}
    if not repeated:
        return texts
    cleaned = []
    for text in texts:
        lines = text.split("\n")
        edge = _edge_indices(lines)
        kept = [
            line for i, line in enumerate(lines)
            if not (i in edge and len(line) <= 80 and _signature(line) in repeated)
        ]
        cleaned.append("\n".join(kept).strip())
    return cleaned


# ----------------------------------------------------------------- loading

def _load_pdf(path: Path, doc_id: str) -> list[Page]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError(f"{path.name} is encrypted, cannot read")
    raw = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw.append(normalize(page.extract_text() or ""))
        except Exception:                       # one bad page must not kill the document
            log.warning("page extraction failed", extra={"source": path.name, "page": number})
            raw.append("")
    texts = strip_repeated_edges(raw)
    pages = [
        Page(doc_id, path.name, number, unwrap_lines(text))
        for number, text in enumerate(texts, start=1) if text.strip()
    ]
    empty = len(raw) - len(pages)
    if empty:
        log.warning("pages with no text (scanned? needs OCR)",
                    extra={"source": path.name, "empty_pages": empty, "total_pages": len(raw)})
    return pages


def _load_docx(path: Path, doc_id: str) -> list[Page]:
    """Word documents. Paragraphs and table cells, in document order.

    Tables matter: a resume or a spec sheet keeps half its facts in them, and
    a loader that reads only paragraphs silently drops that half.
    """
    from docx import Document  # imported lazily, it is optional

    document = Document(str(path))
    parts: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        # Mark headings so the chunker's heading awareness can see them.
        parts.append(f"## {text}" if style.startswith("heading") else text)
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    text = normalize("\n\n".join(parts))
    return [Page(doc_id, path.name, None, text)] if text else []


def _load_csv(path: Path, doc_id: str, rows_per_page: int = 40) -> list[Page]:
    """CSV, as readable records rather than raw comma soup.

    Each row becomes "column: value" lines, because "2024-05-01,impl,done"
    retrieves badly while "status: done" carries its own meaning. Rows are
    batched into pages so a 10k row file does not become one giant chunk.
    """
    import csv as csv_module

    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv_module.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []
        header = [h.strip() or f"column_{i + 1}" for i, h in enumerate(header)]
        records, pages = [], []
        for number, row in enumerate(reader, start=1):
            pairs = [f"{header[i]}: {value.strip()}"
                     for i, value in enumerate(row) if i < len(header) and value.strip()]
            if pairs:
                records.append(f"Row {number}\n" + "\n".join(pairs))
            if len(records) >= rows_per_page:
                pages.append("\n\n".join(records))
                records = []
        if records:
            pages.append("\n\n".join(records))
    return [Page(doc_id, path.name, i, normalize(text))
            for i, text in enumerate(pages, start=1) if text.strip()]


def _load_plain(path: Path, doc_id: str) -> list[Page]:
    text = normalize(path.read_text(encoding="utf-8", errors="replace"))
    return [Page(doc_id, path.name, None, unwrap_lines(text))] if text else []


def load_document(path: str | Path) -> list[Page]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type {suffix!r} for {path.name}")
    doc_id = file_doc_id(path)
    readers = {".pdf": _load_pdf, ".docx": _load_docx, ".csv": _load_csv}
    pages = readers.get(suffix, _load_plain)(path, doc_id)
    log.info("document loaded", extra={"source": path.name, "doc_id": doc_id, "pages": len(pages)})
    return pages
