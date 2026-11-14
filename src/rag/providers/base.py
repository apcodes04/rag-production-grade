"""Provider interfaces (ports) that the pipeline depends on.

Why protocols and not concrete classes: the pipeline must not know whether the
embeddings come from Gemini, OpenAI, a local model or a fake. That single
decision buys offline tests, provider swaps without rewrites, and a cheap way
to reproduce a production bug locally.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from rag.types import LLMResult, Usage


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dim: int
    usage: Usage

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2 normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 L2 normalised vector."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    usage: Usage

    def generate(self, system: str, user: str, *, max_output_tokens: int, temperature: float) -> LLMResult:
        ...


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Make every row unit length so a dot product equals cosine similarity.

    Gemini pre-normalises only the full 3072 dim vectors, so we always do it
    ourselves. The 1e-12 floor avoids dividing by zero on an empty vector.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
