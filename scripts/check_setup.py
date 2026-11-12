"""Sanity check: venv + .env + settings + logging. Run: python scripts/check_setup.py"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.logging_setup import new_request_id, setup_logging
from rag.settings import get_settings

settings = get_settings()
setup_logging(settings.log_level)
log = logging.getLogger("check_setup")

new_request_id()
key = settings.gemini_api_key.get_secret_value()
log.info("settings loaded", extra={
    "key_present": bool(key) and key != "PASTE_YOUR_KEY_HERE",
    "key_length": len(key),
    "chunk_size": settings.chunk_size,
    "chunk_overlap": settings.chunk_overlap,
    "raw_dir_exists": settings.raw_dir.exists(),
})
print("secret printed safely ->", settings.gemini_api_key)
