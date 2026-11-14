"""Client side pacing, so the quota is never exceeded in the first place.

Why this exists even though retries already handle 429:
retrying a rate limit is asking the same question louder. The server is not
broken, it is telling you to slow down, and a retry storm makes the queue
worse for everyone including you. Our 21 question evaluation run proved it:
every answer was eventually correct, but p95 latency went from 3.5 to 11.9
seconds purely from self inflicted 429s.

Token bucket, the standard shape:
  the bucket holds at most `burst` tokens and refills at `rate` per second,
  every request takes one token, and an empty bucket makes the caller wait.
It allows a short burst (good: a few parallel workers start immediately) and
then paces to the long run average (good: the quota is respected).

Thread safe, because the whole point is to coordinate parallel workers.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

log = logging.getLogger(__name__)


class TokenBucket:
    def __init__(self, requests_per_minute: int, burst: int | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.rate = requests_per_minute / 60.0          # tokens per second
        self.capacity = float(burst if burst is not None else max(1, requests_per_minute // 4))
        self._tokens = self.capacity
        self._updated = clock()
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self.waited_s = 0.0                             # for observability
        self.waits = 0

    def _refill(self) -> None:
        now = self._clock()
        # max(0, ...) is load bearing. `_updated` is pushed INTO THE FUTURE when
        # a caller reserves a slot, so a caller arriving before that moment sees
        # a negative elapsed time. Without the clamp it would SUBTRACT tokens,
        # and the waits grow without bound (5s, 15s, 35s, 75s...). See E17.
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._updated = max(self._updated, now)

    def acquire(self) -> float:
        """Take one token, blocking if the bucket is empty. Returns the wait."""
        with self._lock:
            self._refill()
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self.rate
                self.waits += 1
                self.waited_s += wait
            else:
                wait = 0.0
            # Reserve the token now, inside the lock, so parallel callers each
            # wait for their own slot instead of all waking at the same moment.
            self._tokens -= 1.0
            self._updated += wait
        if wait > 0:
            log.debug("rate limit pacing", extra={"sleep_s": round(wait, 3)})
            self._sleep(wait)
        return wait

    def stats(self) -> dict:
        return {"rpm": round(self.rate * 60, 1), "burst": self.capacity,
                "waits": self.waits, "waited_s": round(self.waited_s, 2)}


def build_limiter(requests_per_minute: int) -> TokenBucket | None:
    """0 or less disables pacing, for paid tiers that do not need it."""
    return TokenBucket(requests_per_minute) if requests_per_minute > 0 else None
