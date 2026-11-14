import time

from rag.cache import AnswerCache

CONFIG = {"llm_model": "x", "top_k_final": 5}
BASE = dict(history=[], index_version="idx1", prompt_version="v1", config=CONFIG)


def key(question="How long do refunds take?", **overrides):
    return AnswerCache.make_key(question, **{**BASE, **overrides})


def test_store_and_fetch(tmp_path):
    cache = AnswerCache(tmp_path / "c.sqlite3", ttl_s=60)
    assert cache.get(key()) is None
    cache.set(key(), {"text": "30 days"})
    assert cache.get(key())["text"] == "30 days"
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 1


def test_whitespace_and_case_do_not_create_a_second_entry():
    assert key("How long do REFUNDS take?") == key("  how long do refunds   take? ")


def test_key_changes_when_the_index_is_rebuilt():
    # The reference cached on (question, number of turns), so a reindex kept
    # serving the old answer for an hour. This is the fix.
    assert key() != key(index_version="idx2")


def test_key_changes_with_prompt_version_and_config():
    assert key() != key(prompt_version="v2")
    assert key() != key(config={"llm_model": "y", "top_k_final": 5})


def test_key_changes_with_conversation_history():
    assert key() != key(history=["we were talking about shipping"])


def test_expired_entries_are_not_served(tmp_path):
    cache = AnswerCache(tmp_path / "c.sqlite3", ttl_s=0)
    cache.set(key(), {"text": "stale"})
    time.sleep(0.01)
    assert cache.get(key()) is None
    assert cache.purge_expired() == 1


def test_clear_empties_the_cache(tmp_path):
    cache = AnswerCache(tmp_path / "c.sqlite3", ttl_s=60)
    cache.set(key(), {"text": "x"})
    cache.clear()
    assert cache.stats()["entries"] == 0
