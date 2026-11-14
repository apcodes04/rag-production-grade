"""The LLM call itself, kept deliberately thin.

Everything interesting happens around it: what context goes in (retrieval),
what the prompt says (prompts.py), and whether the answer is trustworthy
(output_guard.py). The generator only has to be robust: a provider can return
no text at all (safety block), or stop mid sentence (token limit), and neither
must be mistaken for a real answer.
"""
from __future__ import annotations

import logging

from rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.providers.base import LLMProvider
from rag.settings import Settings
from rag.types import Candidate, LLMResult, Turn

log = logging.getLogger(__name__)
TRUNCATED_REASONS = ("max_tokens", "maxtokens", "length", "finish_reason.max_tokens")


class Generator:
    def __init__(self, llm: LLMProvider, settings: Settings):
        self.llm = llm
        self.settings = settings

    def generate(self, question: str, candidates: list[Candidate],
                 history: list[Turn] | None = None) -> tuple[LLMResult, str]:
        prompt = build_user_prompt(question, candidates, history, self.settings.history_turns)
        budget = self.settings.llm_max_output_tokens
        result = self.llm.generate(SYSTEM_PROMPT, prompt, max_output_tokens=budget,
                                   temperature=self.settings.llm_temperature)

        if self._was_truncated(result):
            # A truncated answer loses its final citation and often its caveat.
            log.warning("answer truncated by token limit",
                        extra={"finish_reason": result.finish_reason,
                               "thinking_tokens": result.thinking_tokens,
                               "max_output_tokens": budget})
            # Empty AND truncated means a thinking model spent the whole budget
            # reasoning. One retry with double the budget is cheaper than
            # failing the question, and it is bounded: exactly one.
            if self.settings.retry_on_truncation and not result.text.strip():
                log.info("retrying once with a larger output budget",
                         extra={"max_output_tokens": budget * 2})
                result = self.llm.generate(SYSTEM_PROMPT, prompt, max_output_tokens=budget * 2,
                                           temperature=self.settings.llm_temperature)
        return result, prompt

    @staticmethod
    def _was_truncated(result: LLMResult) -> bool:
        reason = str(result.finish_reason).lower()
        return any(flag in reason for flag in TRUNCATED_REASONS)
