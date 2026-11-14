"""Every tunable value in the system, typed and validated at startup.

Rules this file enforces:
1. Secrets never appear in logs (SecretStr).
2. A bad value crashes the process at boot, not during a user request.
3. The same code runs in dev and prod: only environment variables differ.
4. Nothing anywhere else in the codebase hard-codes a model name or a number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore",
    )

    # ---- app
    app_name: str = "rag-from-scratch"
    log_level: str = "INFO"

    # ---- providers: "gemini" talks to Google, "fake" is deterministic and offline
    provider: Literal["gemini", "fake"] = "gemini"
    gemini_api_key: SecretStr | None = None

    # ---- embeddings
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = Field(default=768, gt=0)     # 768 / 1536 / 3072 for this model
    embed_batch_size: int = Field(default=16, gt=0)
    embed_max_retries: int = Field(default=5, ge=0)
    embed_timeout_s: float = Field(default=30.0, gt=0)
    embed_deadline_s: float = Field(default=90.0, gt=0)
    # Client side pacing. Sized to the free tier; 0 disables it. Pacing under
    # the quota beats retrying after a 429, because a retry storm only makes
    # the queue longer.
    embed_requests_per_minute: int = Field(default=60, ge=0)

    # ---- llm
    # gemini-2.0-flash was shut down on 2026-06-01 (the reference project still
    # used it, so it would 404 today). Run scripts/list_models.py to see what
    # your key can actually call, and pin an exact id rather than an alias.
    llm_model: str = "gemini-3.5-flash"
    # Temperature 0 for a retrieval assistant: we want the same context to give
    # the same answer. Newer Gemini models deprecate sampling parameters, so the
    # provider drops them automatically if the API rejects them.
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # This budget covers thinking AND the answer on Gemini 3 models, so it must
    # be generous: at thinking_level high, reasoning expands to fill whatever you
    # give it and the answer can come back completely empty.
    llm_max_output_tokens: int = Field(default=2048, gt=0)
    # Gemini 3 reasons internally before answering. For grounded RAG answers the
    # deep reasoning is mostly wasted latency and cost, so we ask for little.
    # "minimal" | "low" | "medium" | "high" | None (None = do not send the field).
    llm_thinking_level: str | None = "low"
    # A second model to try when the first returns 429 / 503 / timeout. Google
    # serves the flash family from shared capacity, so "currently experiencing
    # high demand" is a normal Tuesday. None disables the failover.
    # Verify the choice with scripts/probe_models.py: being listed by the API
    # does not mean your account may call it (older models are closed to new
    # accounts and answer 404), and an unverified fallback fails exactly when
    # the primary is already down.
    llm_fallback_model: str | None = None
    # If the model burns the whole budget on thinking and returns nothing, try
    # once more with double the budget instead of failing the user's question.
    retry_on_truncation: bool = True
    llm_max_retries: int = Field(default=4, ge=0)
    # One request's patience. A flash model answering from 5 chunks takes a few
    # seconds, so 60 was far too generous: a stuck call burnt a full minute
    # before we learned anything.
    llm_timeout_s: float = Field(default=25.0, gt=0)
    # Patience for the whole call INCLUDING retries and sleeps. This is what
    # makes the worst case predictable, and it is what lets the failover to the
    # second model happen in seconds instead of minutes.
    llm_deadline_s: float = Field(default=70.0, gt=0)
    llm_requests_per_minute: int = Field(default=12, ge=0)

    # ---- chunking (unit: tokens)
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)

    # ---- retrieval
    top_k_dense: int = Field(default=20, gt=0)
    top_k_sparse: int = Field(default=20, gt=0)
    top_k_fused: int = Field(default=20, gt=0)
    top_k_final: int = Field(default=5, gt=0)
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    min_dense_score: float = Field(default=0.30, ge=-1.0, le=1.0)   # refusal gate (cosine)
    min_sparse_score: float = Field(default=2.0, ge=0.0)            # refusal gate (BM25)
    min_candidates: int = Field(default=1, ge=0)
    # When the embedding API is down, answer from BM25 alone instead of failing.
    allow_degraded_retrieval: bool = True
    vector_backend: Literal["numpy", "faiss"] = "numpy"

    # ---- reranker
    reranker: Literal["none", "cross_encoder"] = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_input_k: int = Field(default=20, gt=0)

    # ---- query transformation
    rewrite_followups: bool = True
    history_turns: int = Field(default=6, ge=0)

    # ---- guardrails
    max_query_chars: int = Field(default=2000, gt=0)
    injection_guard: bool = True
    pii_masking: bool = True
    require_citations: bool = True
    min_groundedness: float = Field(default=0.25, ge=0.0, le=1.0)

    # ---- cache
    cache_enabled: bool = True
    cache_ttl_s: int = Field(default=3600, ge=0)

    # ---- paths
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    # Visitor uploads live apart from the seed corpus. raw_dir is committed
    # demo material; upload_dir is whatever a stranger sent us, so it is
    # gitignored and may be wiped on restart without losing anything we own.
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"

    # ---- api
    api_rate_limit_per_minute: int = Field(default=60, gt=0)
    # Empty means no authentication, which is fine on localhost and never
    # anywhere else. The deployed service sets this as a secret.
    api_key: SecretStr | None = None
    # Comma separated. The browser enforces CORS, so this is about which
    # ORIGINS may call us, not about security against a script.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Hosts that deploy per branch (Vercel, Netlify) give every deployment its
    # own subdomain, so an exact allowlist only ever covers production and
    # every preview fails CORS. Keep this scoped to one project: a bare
    # https://.*\.vercel\.app would let any site on that platform call this API.
    # Example: https://rag-from-scratch-[a-z0-9-]+\.vercel\.app
    cors_origin_regex: str = ""
    eval_reports_dir: Path = PROJECT_ROOT / "eval" / "reports"
    # Upload limits. This endpoint is public on the deployed demo, so every
    # one of these is a real control, not a formality.
    max_upload_mb: float = Field(default=10.0, gt=0)
    max_documents: int = Field(default=25, gt=0)
    uploads_per_hour: int = Field(default=20, gt=0)
    # A visitor's upload is theirs, not ours. The browser clears it on leaving;
    # this sweeps anything that outlived its visit, because a closed laptop
    # sends no goodbye.
    upload_ttl_minutes: int = Field(default=60, gt=0)
    # Shown last in the document list: it is the measured corpus, not the
    # thing a visitor should try first.
    eval_corpus_file: str = "AI_QnA_Guide_Extended.pdf"

    def document_roots(self) -> list[Path]:
        """Where documents may live, seed corpus first.

        Order matters: a file present in both roots keys against the seed
        corpus, so a visitor cannot shadow a demo document by uploading a file
        with the same name.
        """
        return [self.raw_dir, self.upload_dir]

    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when provider='gemini'")
        if self.top_k_final > self.top_k_fused:
            raise ValueError("top_k_final cannot exceed top_k_fused")
        return self

    def fingerprint(self) -> dict[str, object]:
        """Config that changes answers. Goes into cache keys and eval reports."""
        return {
            "embedding_model": self.embedding_model, "embedding_dim": self.embedding_dim,
            "llm_model": self.llm_model, "temperature": self.llm_temperature,
            "thinking_level": self.llm_thinking_level,
            "fallback_model": self.llm_fallback_model,
            "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
            "fusion": self.fusion, "rrf_k": self.rrf_k, "dense_weight": self.dense_weight,
            "top_k_dense": self.top_k_dense, "top_k_sparse": self.top_k_sparse,
            "top_k_final": self.top_k_final, "reranker": self.reranker,
            "min_dense_score": self.min_dense_score, "min_sparse_score": self.min_sparse_score,
            "provider": self.provider,
        }


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
