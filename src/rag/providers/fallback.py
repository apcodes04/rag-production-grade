"""Run a second model when the first one is unavailable.

Why this is a separate class and not an `if` inside GeminiLLM: the pipeline
asks for "an LLM", and whether that is one model, two models with a failover,
or a router across five providers is a composition decision. Wrapping keeps
each piece simple and makes the failover testable without touching Gemini.

Only RETRYABLE failures trigger the fallback (429, 503, timeouts). A bad
prompt or a missing model is our bug, and silently asking a second model
would hide it.
"""
from __future__ import annotations

import logging

from rag.errors import RetryableProviderError
from rag.providers.base import LLMProvider
from rag.types import LLMResult, Usage

log = logging.getLogger(__name__)


class FallbackLLM:
    def __init__(self, primary: LLMProvider, secondary: LLMProvider):
        self.primary = primary
        self.secondary = secondary
        self.name = f"{primary.name}+fallback:{secondary.name}"
        self.fallback_calls = 0

    @property
    def usage(self) -> Usage:
        return self.primary.usage.add(self.secondary.usage)

    def generate(self, system: str, user: str, *, max_output_tokens: int,
                 temperature: float) -> LLMResult:
        try:
            return self.primary.generate(system, user, max_output_tokens=max_output_tokens,
                                         temperature=temperature)
        except RetryableProviderError as exc:
            self.fallback_calls += 1
            log.warning("primary model unavailable, falling back",
                        extra={"primary": self.primary.name, "fallback": self.secondary.name,
                               "error": str(exc)[:200]})
            return self.secondary.generate(system, user, max_output_tokens=max_output_tokens,
                                           temperature=temperature)
