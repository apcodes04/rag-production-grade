"""Grader for Lesson 1 (settings). Run: python -m pytest tests/test_settings.py -q"""
import pytest
from pydantic import ValidationError

from rag.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-123")


def test_key_is_loaded_but_hidden():
    s = Settings()
    assert s.gemini_api_key.get_secret_value() == "secret-123"
    assert "secret-123" not in repr(s) and "secret-123" not in str(s)


def test_key_is_required(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)          # ignore the real .env file for this test


def test_defaults():
    s = Settings()
    assert s.chunk_size == 512 and s.chunk_overlap == 64 and s.log_level == "INFO"
    assert s.raw_dir.is_absolute() and s.index_dir.is_absolute()
    assert s.raw_dir.name == "raw" and s.index_dir.name == "index"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "256")
    assert Settings().chunk_size == 256


def test_chunk_size_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(chunk_size=0)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_get_settings_returns_settings():
    assert isinstance(get_settings(), Settings)
