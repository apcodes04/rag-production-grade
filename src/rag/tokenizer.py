"""Text to terms, for BM25 and for lexical checks.

The reference project used `text.lower().split()`, which keeps punctuation
("simple." never matches "simple"), keeps stopwords (every document matches
"the"), and treats "refund" and "refunds" as unrelated. All three quietly
weaken keyword search, and keyword search is the half of hybrid retrieval that
catches names, codes and exact terms.
"""
from __future__ import annotations

import re
import unicodedata

TOKENIZER_VERSION = 1   # bump this when the rules below change: the index must be rebuilt

_TOKEN = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?", re.UNICODE)

STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with without
is are was were be been being am do does did doing have has had having i you he she it we they
them his her its their our your my me us as so such not no nor too very can will just should
what which who whom when where why how about into over under again further once here there all
any both each few more most other some only own same s t don
""".split())


def _fold_plural(token: str) -> str:
    """Very light stemming. Deliberately not Porter: aggressive stemming hurts
    exact term search (the thing BM25 is good at)."""
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True, fold_plurals: bool = True) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = _TOKEN.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if fold_plurals:
        tokens = [_fold_plural(t) for t in tokens]
    return tokens
