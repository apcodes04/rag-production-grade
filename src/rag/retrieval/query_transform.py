"""Turning what the user typed into what we should actually search for.

The bug this fixes (B12 in the audit): the prompt gets the conversation
history, but RETRIEVAL used the raw question. Ask "what about its pricing?"
and the retriever searches for "its pricing", which matches nothing useful.
The fix is to condense the follow up into a standalone question first.

Cost control: an LLM call per query is not free, so a cheap heuristic decides
whether a rewrite is even needed (no history, or no pronoun / ellipsis, means
no rewrite).
"""
from __future__ import annotations

import logging
import re

from rag.providers.base import LLMProvider
from rag.types import Turn

log = logging.getLogger(__name__)

_REFERENTIAL = re.compile(
    r"\b(it|its|it's|they|them|their|this|that|these|those|he|she|his|her|"
    r"the same|above|previous|instead|also|too|what about|how about)\b", re.IGNORECASE)

CONDENSE_SYSTEM = (
    "You rewrite a follow-up question into a standalone question.\n"
    "Rules: keep the user's wording and language where possible, resolve pronouns using the "
    "history, add no new facts, answer nothing. Output ONLY the rewritten question."
)


def needs_rewrite(question: str, history: list[Turn]) -> bool:
    if not history:
        return False
    if len(question.split()) <= 3:               # "and pricing?" style fragments
        return True
    return bool(_REFERENTIAL.search(question))


def condense_question(llm: LLMProvider, question: str, history: list[Turn],
                      *, max_turns: int = 6) -> str:
    recent = history[-max_turns:]
    transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in recent)
    result = llm.generate(
        CONDENSE_SYSTEM, f"History:\n{transcript}\n\nFollow-up question: {question}\n\nStandalone question:",
        max_output_tokens=128, temperature=0.0)
    rewritten = (result.text or "").strip().strip('"')
    # Guard against a chatty or empty model: a rewrite must look like a question,
    # not a paragraph. If it does not, keep the original. Never trust blindly.
    if not rewritten or len(rewritten) > 4 * max(len(question), 80):
        log.warning("query rewrite rejected, using original")
        return question
    return rewritten.split("\n")[0].strip()
