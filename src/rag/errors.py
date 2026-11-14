"""One error family for the whole app.

Why: callers (CLI, API, tests) need to tell apart "user sent something bad"
(4xx, do not retry), "our dependency hiccuped" (retry) and "we are broken"
(5xx, alert). A bare Exception cannot carry that decision.
"""
from __future__ import annotations


class RagError(Exception):
    """Base class. Catch this at the edge of the app."""


class ProviderError(RagError):
    """An external model API failed in a way we should not retry."""


class RetryableProviderError(ProviderError):
    """Rate limit, timeout or 5xx. Safe to retry with backoff."""


class StoreError(RagError):
    """Index on disk is missing, corrupt or incompatible with the settings."""


class GuardrailBlocked(RagError):
    """Input was rejected before any model call."""
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail
