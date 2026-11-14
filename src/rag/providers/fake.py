"""Deterministic offline providers used by every test.

The fake embedder gives every word a fixed random direction (seeded by the
word itself) and averages the directions of the words in the text. Two random
directions in a high dimensional space are nearly orthogonal, so unrelated
texts score near zero while texts sharing words score high: a crude but real
vector space. It costs nothing, never touches the network, and returns byte
identical values on every machine, which is what makes the test suite
trustworthy. A hashed bag of words (one bucket per word) was the obvious
alternative and was rejected: bucket collisions make unrelated texts look
similar, exactly the failure the tests are supposed to catch.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from rag.providers.base import l2_normalise
from rag.types import LLMResult, Usage

_WORD = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=100_000)
def _word_direction(word: str, dim: int) -> np.ndarray:
    seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


class FakeEmbeddings:
    name = "fake-embeddings"

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.usage = Usage()

    def _vector(self, text: str) -> np.ndarray:
        words = _WORD.findall(text.lower())
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        return np.sum([_word_direction(w, self.dim) for w in words], axis=0)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.usage.embed_calls += 1
        self.usage.embed_texts += len(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalise(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        self.usage.embed_calls += 1
        self.usage.embed_texts += 1
        return l2_normalise(self._vector(text))


class FakeLLM:
    """Echoes the first cited source so tests can check grounding and citations."""
    name = "fake-llm"

    def __init__(self, scripted: list[str] | None = None):
        self.usage = Usage()
        self.scripted = list(scripted or [])
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str, *, max_output_tokens: int, temperature: float) -> LLMResult:
        self.calls.append((system, user))
        self.usage.llm_calls += 1
        self.usage.prompt_tokens += len(user) // 4
        text = self.scripted.pop(0) if self.scripted else self._answer_from_context(user)
        self.usage.output_tokens += len(text) // 4
        return LLMResult(text=text, model=self.name, prompt_tokens=len(user) // 4,
                         output_tokens=len(text) // 4)

    @staticmethod
    def _answer_from_context(user: str) -> str:
        match = re.search(r"\[1\][^\n]*\n(.+)", user)
        if not match:
            return "I don't know based on the provided context."
        sentence = re.split(r"(?<=[.!?])\s", match.group(1).strip())[0]
        return f"{sentence} [1]"
