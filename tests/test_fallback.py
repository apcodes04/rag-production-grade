import pytest

from rag.errors import ProviderError, RetryableProviderError
from rag.providers.fallback import FallbackLLM
from rag.types import LLMResult, Usage


class Stub:
    def __init__(self, name, error=None, text="answer"):
        self.name = name
        self.error = error
        self.text = text
        self.calls = 0
        self.usage = Usage()

    def generate(self, system, user, *, max_output_tokens, temperature):
        self.calls += 1
        if self.error:
            raise self.error
        self.usage.llm_calls += 1
        self.usage.output_tokens += 10
        return LLMResult(self.text, self.name)


def run(primary, secondary):
    return FallbackLLM(primary, secondary).generate("s", "u", max_output_tokens=10,
                                                    temperature=0.0)


def test_the_primary_is_used_when_it_works():
    primary, secondary = Stub("a"), Stub("b")
    assert run(primary, secondary).model == "a"
    assert secondary.calls == 0


def test_a_503_moves_to_the_fallback():
    primary = Stub("a", error=RetryableProviderError("503 UNAVAILABLE"))
    secondary = Stub("b")
    assert run(primary, secondary).model == "b"
    assert secondary.calls == 1


def test_a_real_bug_is_not_hidden_by_the_fallback():
    # A bad request or an unknown model is our mistake. Asking a second model
    # would turn a loud bug into a silent cost.
    primary = Stub("a", error=ProviderError("404 model not found"))
    secondary = Stub("b")
    with pytest.raises(ProviderError):
        run(primary, secondary)
    assert secondary.calls == 0


def test_usage_covers_both_models():
    primary, secondary = Stub("a"), Stub("b")
    llm = FallbackLLM(primary, secondary)
    llm.generate("s", "u", max_output_tokens=10, temperature=0.0)
    assert llm.usage.llm_calls == 1 and llm.usage.output_tokens == 10
    assert "fallback" in llm.name


# ---- retry budget: bounded by attempts AND by the clock

def test_retry_stops_at_the_deadline_even_with_attempts_left(monkeypatch, caplog):
    import logging
    import time as time_module

    from rag.errors import RetryableProviderError
    from rag.providers import gemini

    slept: list[float] = []
    clock = {"now": 0.0}
    monkeypatch.setattr(gemini.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(gemini.time, "sleep",
                        lambda s: (slept.append(s), clock.__setitem__("now", clock["now"] + 30)))
    calls = {"n": 0}

    def always_timeout():
        calls["n"] += 1
        clock["now"] += 25                      # each attempt burns its own timeout
        raise TimeoutError("504 DEADLINE_EXCEEDED")

    with caplog.at_level(logging.WARNING), pytest.raises(RetryableProviderError):
        gemini.with_retry(always_timeout, attempts=10, what="generate_content", deadline_s=70)

    assert calls["n"] < 11                      # the attempt budget was never reached
    assert "retry budget exhausted" in caplog.text
    assert time_module is not None


def test_without_a_deadline_the_attempt_count_still_bounds_it(monkeypatch):
    from rag.errors import RetryableProviderError
    from rag.providers import gemini

    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_503():
        calls["n"] += 1
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(RetryableProviderError):
        gemini.with_retry(always_503, attempts=3, what="x")
    assert calls["n"] == 4                      # the first try plus three retries
