"""Answer cache, keyed on everything that can change the answer.

The reference project cached on (question, number of turns so far), which is
wrong in two directions: two different conversations with the same turn count
collide, and a reindex or a prompt change keeps serving yesterday's answer.

The key here is a hash of: normalised question, conversation fingerprint,
index version, prompt version, and the config fingerprint (model, top_k,
reranker, thresholds). Change any of them and the cache misses, which is
exactly what you want.

sqlite rather than a library: one file, no server, transactional, already in
the standard library. Redis is the swap when you have more than one process.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
  key TEXT PRIMARY KEY, payload TEXT NOT NULL,
  created_at REAL NOT NULL, expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS answers_expiry ON answers(expires_at);
"""


class AnswerCache:
    def __init__(self, path: Path, ttl_s: int = 3600):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(question: str, *, history: list[str], index_version: str,
                 prompt_version: str, config: dict[str, Any],
                 sources: list[str] | None = None) -> str:
        material = json.dumps({
            "q": " ".join(question.lower().split()),
            "history": [" ".join(turn.lower().split()) for turn in history],
            "index": index_version, "prompt": prompt_version, "config": config,
            # The document filter changes the answer, so it has to change the
            # key. Without it, asking one question against one document and
            # then against the whole corpus returns the first answer twice.
            "sources": sorted(sources) if sources else None,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        row = self._db.execute(
            "SELECT payload, expires_at FROM answers WHERE key = ?", (key,)).fetchone()
        if not row or row[1] < time.time():
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def set(self, key: str, payload: dict) -> None:
        now = time.time()
        self._db.execute(
            "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)",
            (key, json.dumps(payload, ensure_ascii=False), now, now + self.ttl_s))
        self._db.commit()

    def purge_expired(self) -> int:
        cursor = self._db.execute("DELETE FROM answers WHERE expires_at < ?", (time.time(),))
        self._db.commit()
        return cursor.rowcount

    def stats(self) -> dict:
        size = self._db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        total = self.hits + self.misses
        return {"entries": size, "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}

    def clear(self) -> None:
        self._db.execute("DELETE FROM answers")
        self._db.commit()
