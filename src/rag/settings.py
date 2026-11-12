"""Central, typed, validated settings.

Why this exists: every tunable lives in ONE place, is type checked at startup,
and secrets never leak into logs. Reference code used a plain dataclass and
hard-coded the embedding model in a second file (config drift).
Settings grow module by module: we add a field only when a module needs it.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",   # absolute path: works from any cwd
        env_file_encoding="utf-8",
        extra="ignore",                   # unknown env vars are not errors
    )

    # Secrets (SecretStr: prints as '**********', never leaks in logs)
    gemini_api_key: SecretStr

    # Paths
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    index_dir: Path = PROJECT_ROOT / "data" / "index"

    # Chunking (unit = tokens, decided in the chunking module)
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)

    # Logging
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


def get_settings() -> Settings:
    return Settings()  # reads env vars first, then .env
