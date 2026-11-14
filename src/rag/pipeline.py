"""The orchestrator: one question in, one checked and timed Answer out.

Order of operations, and why each step is where it is:

  1. input guard      cheapest check first, and it runs before we spend a
                      single token on a hostile or empty query
  2. query rewrite    retrieval must search for a standalone question, not
                      "what about its price?"
  3. cache lookup     after the rewrite, so two phrasings of the same
                      follow up share a cache entry
  4. hybrid retrieval dense plus sparse, fused
  5. relevance gate   if nothing is relevant we refuse HERE, before the LLM.
                      This single branch removes most hallucinations.
  6. rerank           expensive, so only on the survivors
  7. generate         the actual LLM call
  8. output guard     citations must be real and the answer must be grounded
  9. cache store      only grounded, non refused answers are ever cached

Every stage is timed and logged under one request id, so a slow or wrong
answer can be traced without reproducing it.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from rag.cache import AnswerCache
from rag.generation import Generator
from rag.guardrails.input_guard import check_query, mask_pii, scan_documents
from rag.guardrails.output_guard import check_answer
from rag.logging_setup import new_request_id
from rag.prompts import PROMPT_VERSION, REFUSAL_MESSAGE
from rag.providers.base import EmbeddingProvider, LLMProvider
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.query_transform import condense_question, needs_rewrite
from rag.retrieval.reranker import Reranker, build_reranker
from rag.settings import Settings
from rag.stores.index import Index
from rag.types import Answer, Candidate, Turn, Usage

log = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings, index: Index, embedder: EmbeddingProvider,
                 llm: LLMProvider, reranker: Reranker | None = None,
                 cache: AnswerCache | None = None):
        self.settings = settings
        self.index = index
        self.embedder = embedder
        self.llm = llm
        self.retriever = HybridRetriever(index, embedder, settings)
        self.reranker = reranker or build_reranker(settings)
        self.generator = Generator(llm, settings)
        self.cache = cache

    # ------------------------------------------------------------- helpers
    @contextmanager
    def _stage(self, timings: dict[str, float], name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = round((time.perf_counter() - start) * 1000, 1)

    def _usage_since(self, before: tuple[Usage, Usage]) -> Usage:
        embed_before, llm_before = before
        embed, llm = self.embedder.usage, self.llm.usage
        return Usage(
            embed_calls=embed.embed_calls - embed_before.embed_calls,
            embed_texts=embed.embed_texts - embed_before.embed_texts,
            llm_calls=llm.llm_calls - llm_before.llm_calls,
            prompt_tokens=llm.prompt_tokens - llm_before.prompt_tokens,
            output_tokens=llm.output_tokens - llm_before.output_tokens,
            thinking_tokens=llm.thinking_tokens - llm_before.thinking_tokens)

    # -------------------------------------------------------------- answer
    def answer(self, question: str, history: list[Turn] | None = None,
               *, use_cache: bool = True, sources: list[str] | None = None) -> Answer:
        """Answer one question, optionally restricted to named documents.

        `sources` narrows retrieval to those documents only. It is part of the
        cache key, because the same question against a different set of
        documents is a different question.
        """
        history = history or []
        request_id = new_request_id()
        timings: dict[str, float] = {}
        usage_before = (Usage(**vars(self.embedder.usage)), Usage(**vars(self.llm.usage)))
        log.info("question received", extra={"question": mask_pii(question),
                                             "history_turns": len(history)})

        def finish(answer: Answer) -> Answer:
            answer.request_id = request_id
            answer.timings_ms = timings
            answer.usage = self._usage_since(usage_before)
            log.info("answered", extra={"refused": answer.refused,
                                        "reason": answer.refusal_reason,
                                        "cache_hit": answer.cache_hit,
                                        "citations": answer.citations,
                                        "groundedness": answer.groundedness,
                                        "timings_ms": timings,
                                        "usage": vars(answer.usage)})
            return answer

        # 1. input guard
        with self._stage(timings, "input_guard"):
            verdict = check_query(question, max_chars=self.settings.max_query_chars,
                                  injection_guard=self.settings.injection_guard)
        if not verdict.allowed:
            log.warning("query blocked", extra={"reason": verdict.reason, "flags": verdict.flags})
            return finish(Answer(question=question, text=_blocked_message(verdict.reason),
                                 refused=True, refusal_reason=verdict.reason))
        clean_question = verdict.query

        # 2. query rewrite for follow ups
        search_query, rewritten = clean_question, None
        if self.settings.rewrite_followups and needs_rewrite(clean_question, history):
            with self._stage(timings, "rewrite"):
                search_query = condense_question(self.llm, clean_question, history,
                                                 max_turns=self.settings.history_turns)
            rewritten = search_query if search_query != clean_question else None

        # 3. cache
        cache_key = None
        if self.cache and use_cache and self.settings.cache_enabled:
            cache_key = AnswerCache.make_key(
                search_query, history=[t.content for t in history[-self.settings.history_turns:]],
                index_version=self.index.index_version(), prompt_version=PROMPT_VERSION,
                config=self.settings.fingerprint(), sources=sources)
            with self._stage(timings, "cache_lookup"):
                cached = self.cache.get(cache_key)
            if cached:
                return finish(Answer(
                    question=question, text=cached["text"], citations=cached["citations"],
                    groundedness=cached.get("groundedness"), cache_hit=True,
                    rewritten_question=rewritten,
                    sources=[Candidate(chunk=self.index.chunks[cid])
                             for cid in cached.get("chunk_ids", []) if cid in self.index.chunks]))

        # 4. retrieval
        with self._stage(timings, "retrieval"):
            retrieved = self.retriever.retrieve(search_query, sources=sources)
        candidates = retrieved.candidates

        # 5. relevance gate: refuse before spending an LLM call
        if not self.retriever.is_relevant_enough(candidates):
            log.info("no relevant context", extra={"candidates": len(candidates)})
            return finish(Answer(question=question, text=REFUSAL_MESSAGE, refused=True,
                                 refusal_reason="no_relevant_context",
                                 rewritten_question=rewritten, sources=candidates[:3],
                                 degraded=retrieved.degraded,
                                 degraded_reason=retrieved.degraded_reason))

        # 6. rerank
        with self._stage(timings, "rerank"):
            top = self.reranker.rerank(search_query, candidates[: self.settings.rerank_input_k],
                                       self.settings.top_k_final)

        suspicious = scan_documents([c.chunk.text for c in top])
        if suspicious:
            # Kept, not dropped: a document may legitimately discuss injection.
            # The prompt already marks the context as untrusted data.
            log.warning("injection-like text inside retrieved context",
                        extra={"positions": suspicious,
                               "chunk_ids": [top[i].chunk.chunk_id for i in suspicious]})

        # 7. generate
        with self._stage(timings, "generate"):
            result, _prompt = self.generator.generate(search_query, top, history)

        # 8. output guard
        with self._stage(timings, "output_guard"):
            checked = check_answer(result.text, [c.chunk.text for c in top],
                                   require_citations=self.settings.require_citations,
                                   min_groundedness=self.settings.min_groundedness)
        if not checked.ok:
            log.warning("answer rejected by output guard",
                        extra={"reason": checked.reason, "groundedness": checked.groundedness,
                               "invalid_citations": checked.invalid_citations,
                               "draft": result.text[:300]})
            return finish(Answer(question=question, text=REFUSAL_MESSAGE, refused=True,
                                 refusal_reason=checked.reason, sources=top,
                                 groundedness=checked.groundedness,
                                 rewritten_question=rewritten))

        answer = Answer(question=question, text=result.text, sources=top,
                        citations=checked.citations, groundedness=checked.groundedness,
                        rewritten_question=rewritten, degraded=retrieved.degraded,
                        degraded_reason=retrieved.degraded_reason)
        if checked.reason == "model_refused":
            # The retrieval gate let this through, but the model itself said the
            # context does not answer the question. That IS a refusal, and if we
            # do not record it as one, every refusal metric is wrong: the system
            # behaves correctly while the dashboard says it does not.
            answer.refused = True
            answer.refusal_reason = "model_refused"

        # 9. cache only what is worth replaying
        # A degraded answer must never be cached: it would outlive the outage.
        if cache_key and self.cache and not retrieved.degraded:
            self.cache.set(cache_key, {"text": answer.text, "citations": answer.citations,
                                       "groundedness": answer.groundedness,
                                       "chunk_ids": [c.chunk.chunk_id for c in top]})
        return finish(answer)


def _blocked_message(reason: str | None) -> str:
    return {
        "empty_query": "Please type a question.",
        "query_too_long": "That question is too long. Please shorten it.",
        "prompt_injection": "That request cannot be processed.",
    }.get(reason or "", "That request cannot be processed.")
