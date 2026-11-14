import logging

from rag.generation import Generator
from rag.types import Candidate, Chunk, LLMResult, Usage


class StubLLM:
    """Returns the scripted results in order and records the budget it was given."""
    name = "stub"

    def __init__(self, results):
        self.results = list(results)
        self.budgets: list[int] = []
        self.usage = Usage()

    def generate(self, system, user, *, max_output_tokens, temperature):
        self.budgets.append(max_output_tokens)
        return self.results.pop(0)


def candidates():
    return [Candidate(chunk=Chunk("c1", "d", "a.txt", 1, 0, "Refunds take 30 days.", 5))]


def test_a_thinking_model_that_ate_the_budget_is_retried_once(settings, caplog):
    llm = StubLLM([LLMResult("", "m", finish_reason="FinishReason.MAX_TOKENS", thinking_tokens=2048),
                   LLMResult("Refunds take 30 days [1].", "m")])
    with caplog.at_level(logging.INFO):
        result, _ = Generator(llm, settings).generate("how long?", candidates())
    assert result.text == "Refunds take 30 days [1]."
    assert llm.budgets == [settings.llm_max_output_tokens, settings.llm_max_output_tokens * 2]
    assert "retrying once with a larger output budget" in caplog.text


def test_the_retry_happens_at_most_once(settings):
    llm = StubLLM([LLMResult("", "m", finish_reason="MAX_TOKENS"),
                   LLMResult("", "m", finish_reason="MAX_TOKENS")])
    result, _ = Generator(llm, settings).generate("how long?", candidates())
    assert result.text == "" and len(llm.budgets) == 2        # not three, not a loop


def test_a_truncated_but_non_empty_answer_is_not_retried(settings, caplog):
    llm = StubLLM([LLMResult("Refunds take 30 d", "m", finish_reason="MAX_TOKENS")])
    with caplog.at_level(logging.WARNING):
        Generator(llm, settings).generate("how long?", candidates())
    assert len(llm.budgets) == 1
    assert "answer truncated by token limit" in caplog.text


def test_the_retry_can_be_switched_off(settings):
    llm = StubLLM([LLMResult("", "m", finish_reason="MAX_TOKENS")])
    Generator(llm, settings.model_copy(update={"retry_on_truncation": False})).generate(
        "how long?", candidates())
    assert len(llm.budgets) == 1
