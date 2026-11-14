"""The limiter is tested on a FAKE clock: a test that really sleeps is a test
nobody runs."""
import threading

import pytest

from rag.providers.rate_limit import TokenBucket, build_limiter


class Clock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def bucket(rpm=60, burst=None):
    clock = Clock()
    return TokenBucket(rpm, burst=burst, clock=clock.time, sleep=clock.sleep), clock


def test_a_burst_goes_through_without_waiting():
    limiter, clock = bucket(rpm=60, burst=5)
    waits = [limiter.acquire() for _ in range(5)]
    assert waits == [0.0] * 5 and clock.slept == []


def test_after_the_burst_requests_are_paced_to_the_rate():
    limiter, clock = bucket(rpm=60, burst=2)        # 1 request per second
    limiter.acquire()
    limiter.acquire()
    assert limiter.acquire() == pytest.approx(1.0)  # third one waits a second
    assert clock.slept == [pytest.approx(1.0)]


def test_tokens_refill_while_the_caller_is_idle():
    limiter, clock = bucket(rpm=60, burst=2)
    limiter.acquire()
    limiter.acquire()
    clock.now += 5                                  # idle for five seconds
    assert limiter.acquire() == 0.0                 # bucket refilled, no wait


def test_parallel_callers_get_their_own_slot_not_the_same_one():
    limiter, _clock = bucket(rpm=60, burst=1)
    waits: list[float] = []
    lock = threading.Lock()

    def worker():
        wait = limiter.acquire()
        with lock:
            waits.append(wait)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # One free slot, then three callers queued behind each other: the waits
    # must be distinct and increasing, never all the same value.
    assert sorted(waits) == pytest.approx([0.0, 1.0, 2.0, 3.0], abs=0.01)


def test_stats_expose_the_cost_of_pacing():
    limiter, _ = bucket(rpm=60, burst=1)
    limiter.acquire()
    limiter.acquire()
    stats = limiter.stats()
    assert stats["waits"] == 1 and stats["waited_s"] > 0 and stats["rpm"] == 60.0


def test_zero_disables_pacing_and_negative_is_rejected():
    assert build_limiter(0) is None
    assert isinstance(build_limiter(10), TokenBucket)
    with pytest.raises(ValueError):
        TokenBucket(0)


class FrozenClock:
    """A clock where sleeping does NOT advance time.

    This is the honest model of parallel callers: three threads sleeping at the
    same moment overlap in wall time, they do not queue up in it. The first
    version of these tests used a clock that advanced on sleep, which is what
    let the unbounded-wait bug (E17) through.
    """
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def test_waits_stay_bounded_when_callers_sleep_in_parallel():
    clock = FrozenClock()
    limiter = TokenBucket(12, burst=3, clock=clock.time, sleep=clock.sleep)  # 1 per 5s
    waits = [limiter.acquire() for _ in range(7)]
    assert waits[:3] == [0.0, 0.0, 0.0]                   # the burst
    # Each later caller takes the NEXT free slot: 5, 10, 15, 20 seconds.
    assert waits[3:] == pytest.approx([5.0, 10.0, 15.0, 20.0])
    # The bug produced 5, 15, 35, 75: each wait roughly doubling.
    assert max(waits) < 25


def test_a_long_idle_period_still_refills_with_the_clamp():
    clock = FrozenClock()
    limiter = TokenBucket(60, burst=2, clock=clock.time, sleep=clock.sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()                                      # reserves a future slot
    clock.now += 600                                       # ten idle minutes
    assert limiter.acquire() == 0.0                        # bucket is full again
