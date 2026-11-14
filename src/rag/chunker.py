"""Structure-aware, token-budgeted chunker.

Strategy (recursive): try the coarsest boundary first (paragraph), and only
when a piece is still too big go finer: line -> sentence -> word -> raw slice.
Then greedily pack small pieces together up to chunk_size, carrying overlap
made of WHOLE units (never cut mid-sentence) and keeping headings glued to
the text that follows them.

Fixes vs reference: a hard token cap that is really enforced (B1), nothing is
dropped for being short (B2), deterministic unique chunk ids (B6), the unit is
tokens not words (B9).
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable

from rag.types import Chunk, Page

Unit = tuple[str, str]   # (separator that preceded this unit, text)

_PIECE = re.compile(r"\w+|[^\w\s]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_NOT_HEADING_END = (".", "!", "?", ":", ";", ",")
_HEADING_MAX_TOKENS = 12


def count_tokens(text: str) -> int:
    """Approximate token count (no model tokenizer needed, deterministic).

    Each word or punctuation mark is 1 token, long words cost more (about 6
    chars per token), then +10% safety margin. Slightly conservative on
    purpose: chunks stay under the real limit. Swap for a real tokenizer later.
    """
    if not text.strip():
        return 0
    pieces = sum(max(1, math.ceil(len(p) / 6)) for p in _PIECE.findall(text))
    return math.ceil(pieces * 1.1)


_SPLITTERS: list[tuple[Callable[[str], list[str]], str]] = [
    (lambda t: t.split("\n\n"), "\n\n"),   # paragraphs
    (lambda t: t.split("\n"), "\n"),       # lines (lists, table rows, headings)
    (lambda t: _SENTENCE.split(t), " "),   # sentences
    (lambda t: t.split(" "), " "),         # words
]


def _hard_slice(text: str, max_tokens: int, lead: str) -> list[Unit]:
    """Last resort for a single 'word' bigger than the budget (URL, base64, ...)."""
    size = max(1, max_tokens * 5)
    return [(lead if i == 0 else "", text[i:i + size]) for i in range(0, len(text), size)]


def _atomize(text: str, max_tokens: int, level: int = 0, lead: str = "") -> list[Unit]:
    """Break text into units that each fit max_tokens, using the coarsest boundary that works."""
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [(lead, text)]
    if level >= len(_SPLITTERS):
        return _hard_slice(text, max_tokens, lead)
    splitter, sep = _SPLITTERS[level]
    parts = [p.strip() for p in splitter(text) if p.strip()]
    if len(parts) <= 1:
        return _atomize(text, max_tokens, level + 1, lead)
    units: list[Unit] = []
    for i, part in enumerate(parts):
        units.extend(_atomize(part, max_tokens, level + 1, lead if i == 0 else sep))
    return units


def _join(units: list[Unit]) -> str:
    return "".join((sep if i else "") + text for i, (sep, text) in enumerate(units))


def _tokens(units: list[Unit]) -> int:
    return sum(count_tokens(text) for _, text in units)


def _is_heading(unit: Unit) -> bool:
    text = unit[1]
    return (count_tokens(text) <= _HEADING_MAX_TOKENS
            and "\n" not in text and not text.endswith(_NOT_HEADING_END))


def _tail(units: list[Unit], overlap: int) -> list[Unit]:
    """Trailing whole units worth at most `overlap` tokens."""
    tail: list[Unit] = []
    total = 0
    for unit in reversed(units):
        t = count_tokens(unit[1])
        if total + t > overlap:
            break
        tail.insert(0, unit)
        total += t
    return tail


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[str] = []
    cur: list[Unit] = []
    cur_tokens = 0
    for unit in _atomize(text, chunk_size):
        t = count_tokens(unit[1])
        if cur and cur_tokens + t > chunk_size:
            carry: list[Unit] = []                 # never end a chunk on a heading
            while len(cur) > 1 and _is_heading(cur[-1]):
                carry.insert(0, cur.pop())
            chunks.append(_join(cur))
            cur = _tail(cur, overlap) + carry
            if _tokens(cur) + t > chunk_size:      # overlap does not fit, drop it
                cur = carry
                if cur and _tokens(cur) + t > chunk_size:
                    chunks.append(_join(cur))      # heading alone, but never lose text
                    cur = []
            cur_tokens = _tokens(cur)
        cur.append(unit)
        cur_tokens += t
    if cur:
        chunks.append(_join(cur))
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for page in pages:
        for text in split_text(page.text, chunk_size, overlap):
            index = counters.get(page.doc_id, 0)
            counters[page.doc_id] = index + 1
            chunks.append(Chunk(
                chunk_id=f"{page.doc_id}-{index:05d}",
                doc_id=page.doc_id, source=page.source, page=page.page,
                chunk_index=index, text=text, token_count=count_tokens(text),
            ))
    return chunks
