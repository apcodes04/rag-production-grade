"""Google Gemini embeddings and generation, wrapped for production use.

What the wrapper adds over calling the SDK directly:
  retries with exponential backoff and jitter on 429 / 5xx / timeout,
  batching with a per batch fallback when the model rejects batches,
  task_type (documents and queries are embedded differently, this matters),
  dimension validation (a silent dim change would corrupt the whole index),
  explicit handling of empty responses (safety blocks return no text),
  token usage accounting so cost is visible.
"""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from rag.errors import ProviderError, RetryableProviderError
from rag.providers.base import l2_normalise
from rag.providers.rate_limit import TokenBucket
from rag.types import LLMResult, Usage

log = logging.getLogger(__name__)
T = TypeVar("T")

_RETRYABLE_CODES = {408, 409, 429, 500, 502, 503, 504}
_RETRYABLE_WORDS = ("rate limit", "resource_exhausted", "unavailable", "deadline",
                    "timeout", "timed out", "internal error", "overloaded")


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_CODES:
        return True
    message = str(exc).lower()
    return any(word in message for word in _RETRYABLE_WORDS)


def with_retry(fn: Callable[[], T], *, attempts: int, what: str, base_delay: float = 0.5,
               deadline_s: float | None = None) -> T:
    """Retry `fn` on transient failures, bounded by BOTH a count and a clock.

    The count alone is not enough. With a 60 second timeout and 4 retries, one
    stuck request blocks for five minutes before anyone finds out, and a batch
    job of 13 questions can take an hour. A wall clock deadline is what makes
    the worst case predictable, which is the whole point of a retry policy.
    """
    last: Exception | None = None
    started = time.monotonic()
    for attempt in range(attempts + 1):
        try:
            return fn()
        except Exception as exc:                      # noqa: BLE001 SDK raises many types
            last = exc
            if not _is_retryable(exc) or attempt == attempts:
                break
            delay = min(base_delay * (2 ** attempt), 8.0)
            delay += random.uniform(0, delay * 0.3)
            elapsed = time.monotonic() - started
            if deadline_s is not None and elapsed + delay >= deadline_s:
                log.warning("retry budget exhausted, giving up early", extra={
                    "what": what, "attempt": attempt + 1, "elapsed_s": round(elapsed, 1),
                    "deadline_s": deadline_s})
                break
            log.warning("retrying provider call", extra={
                "what": what, "attempt": attempt + 1, "sleep_s": round(delay, 2),
                "elapsed_s": round(elapsed, 1), "error": str(exc)[:200]})
            time.sleep(delay)
    assert last is not None
    if _is_retryable(last):
        raise RetryableProviderError(f"{what} failed after {attempts} retries: {last}") from last
    raise ProviderError(f"{what} failed: {last}") from last


def _client(api_key: str, timeout_s: float):
    from google import genai
    from google.genai import types as genai_types
    try:    # timeout is in milliseconds; older SDKs do not accept http_options
        return genai.Client(api_key=api_key,
                            http_options=genai_types.HttpOptions(timeout=int(timeout_s * 1000)))
    except TypeError:
        return genai.Client(api_key=api_key)


class GeminiEmbeddings:
    def __init__(self, api_key: str, model: str, dim: int, *, batch_size: int = 16,
                 max_retries: int = 5, timeout_s: float = 60.0, deadline_s: float = 90.0,
                 limiter: TokenBucket | None = None):
        self.name = model
        self.dim = dim
        self.usage = Usage()
        self._model = model
        self._batch_size = batch_size
        self._retries = max_retries
        self._deadline = deadline_s
        self._client = _client(api_key, timeout_s)
        self._limiter = limiter
        self._batch_supported = True

    def _config(self, task_type: str):
        from google.genai import types as genai_types
        return genai_types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)

    def _call(self, contents: list[str], task_type: str) -> list[list[float]]:
        def run():
            if self._limiter:
                self._limiter.acquire()          # pace BEFORE the call, not after a 429
            response = self._client.models.embed_content(
                model=self._model, contents=contents, config=self._config(task_type))
            return [embedding.values for embedding in response.embeddings]

        self.usage.embed_calls += 1
        self.usage.embed_texts += len(contents)
        return with_retry(run, attempts=self._retries, what="embed_content",
                          deadline_s=self._deadline)

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size if self._batch_supported else 1):
            batch = texts[start: start + (self._batch_size if self._batch_supported else 1)]
            try:
                vectors.extend(self._call(batch, task_type))
            except ProviderError:
                if not self._batch_supported or len(batch) == 1:
                    raise
                # Some models accept only one input per request. Degrade, do not die.
                log.warning("batch embedding rejected, falling back to one call per text")
                self._batch_supported = False
                for text in batch:
                    vectors.extend(self._call([text], task_type))
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape[1] != self.dim:
            raise ProviderError(
                f"embedding dim mismatch: model returned {array.shape[1]}, settings say {self.dim}")
        return l2_normalise(array)      # required for any dim other than 3072

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


# Rough token equivalents of the thinking levels, used only on SDKs too old to
# know about thinking_level (thinking_budget is the documented fallback path).
_LEVEL_TO_BUDGET = {"minimal": 0, "low": 512, "medium": 2048, "high": -1}


def _build_thinking_config(level: str | None):
    """Gemini 3 takes `thinking_level`, Gemini 2.5 and older SDKs take
    `thinking_budget`, and sending both to a Gemini 3 model is an error. Build
    whichever one this SDK understands, and nothing if it understands neither."""
    if not level:
        return None
    from google.genai import types as genai_types
    for kwargs in ({"thinking_level": level},
                   {"thinking_budget": _LEVEL_TO_BUDGET.get(level, 512)}):
        try:
            return genai_types.ThinkingConfig(**kwargs)
        except Exception:                    # noqa: BLE001 field missing in this SDK version
            continue
    log.warning("this google-genai version supports no thinking config, upgrade to >=1.51")
    return None


class GeminiLLM:
    def __init__(self, api_key: str, model: str, *, max_retries: int = 4, timeout_s: float = 60.0,
                 thinking_level: str | None = "low", deadline_s: float = 90.0,
                 limiter: TokenBucket | None = None):
        self.name = model
        self.usage = Usage()
        self._model = model
        self._retries = max_retries
        self._deadline = deadline_s
        self._client = _client(api_key, timeout_s)
        self._limiter = limiter
        self._send_sampling = True      # turned off if the API rejects temperature
        self._thinking = _build_thinking_config(thinking_level)

    def generate(self, system: str, user: str, *, max_output_tokens: int,
                 temperature: float) -> LLMResult:
        from google.genai import types as genai_types

        def run():
            if self._limiter:
                self._limiter.acquire()
            options = {"system_instruction": system, "max_output_tokens": max_output_tokens}
            if self._send_sampling:
                options["temperature"] = temperature
            if self._thinking is not None:
                options["thinking_config"] = self._thinking
            return self._client.models.generate_content(
                model=self._model, contents=user,
                config=genai_types.GenerateContentConfig(**options))

        try:
            response = with_retry(run, attempts=self._retries, what="generate_content",
                                  deadline_s=self._deadline)
        except ProviderError as exc:
            # Gemini 3.x deprecates temperature / top_p / top_k. Drop them once
            # and retry rather than failing the request over a config field.
            if not self._send_sampling or "temperature" not in str(exc).lower():
                raise
            log.warning("model rejected sampling parameters, retrying without them")
            self._send_sampling = False
            response = with_retry(run, attempts=self._retries, what="generate_content",
                                  deadline_s=self._deadline)
        text = getattr(response, "text", None) or ""
        finish = "stop"
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = str(getattr(candidates[0], "finish_reason", "stop") or "stop")
        if not text:
            # Safety block, recitation block or zero candidates. Never crash here:
            # the pipeline turns an empty answer into an honest refusal.
            finish = finish if candidates else "blocked"
            # Three different causes, and the log must say which one: MAX_TOKENS
            # with thinking tokens spent means the model reasoned until the
            # budget ran out. Otherwise it is a safety block or zero candidates.
            log.warning("llm returned no text", extra={
                "finish_reason": finish,
                "thinking_tokens": int(getattr(
                    getattr(response, "usage_metadata", None), "thoughts_token_count", 0) or 0),
                "hint": "raise llm_max_output_tokens or lower llm_thinking_level"})
        meta: Any = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(meta, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(meta, "candidates_token_count", 0) or 0)
        thinking_tokens = int(getattr(meta, "thoughts_token_count", 0) or 0)
        self.usage.llm_calls += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.output_tokens += output_tokens
        self.usage.thinking_tokens += thinking_tokens
        return LLMResult(text=text, model=self._model, finish_reason=finish,
                         prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                         thinking_tokens=thinking_tokens)
