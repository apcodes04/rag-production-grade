"""Upload endpoint tests.

This endpoint is the only one a stranger can use to spend our money and fill
our disk, so most of these tests are about what it REFUSES, not what it accepts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def api(monkeypatch, pipeline):
    import rag.api as api_module
    monkeypatch.setattr(api_module, "_pipeline", pipeline)
    monkeypatch.setattr(api_module.settings, "api_key", None)
    monkeypatch.setattr(api_module.settings, "uploads_per_hour", 50)
    monkeypatch.setattr(api_module.settings, "api_rate_limit_per_minute", 500)
    api_module._calls.clear()
    api_module._uploads.clear()
    return api_module, TestClient(api_module.app)


def upload(client, name: str, content: bytes):
    return client.post("/documents", files={"file": (name, content, "application/octet-stream")})


# ------------------------------------------------------------- happy path

def test_upload_indexes_the_file_and_reports_what_it_did(api):
    api_module, client = api
    before = len(api_module._pipeline.index.chunks)
    body = upload(client, "handbook.txt",
                  b"Leave policy. Employees get twenty five days of paid leave each year. "
                  b"Unused leave carries over for one quarter only.").json()

    assert body["status"] == "added"
    assert body["chunks"] > 0 and body["tokens"] > 0
    assert body["total_chunks"] == before + body["chunks"]
    assert body["sample_chunks"][0]["text"]
    assert body["ingest_ms"] >= 0


def test_an_uploaded_document_is_immediately_answerable(api):
    """The whole point of the upload page: ask about it on the very next call."""
    _, client = api
    upload(client, "perks.txt",
           b"Perks. The company reimburses home office equipment up to one lakh rupees. "
           b"Claims are filed through the finance portal within sixty days.")
    body = client.post("/ask", json={"question": "What is the home office reimbursement?"}).json()
    assert body["refused"] is False
    assert any(s["source"] == "perks.txt" for s in body["sources"])


def test_re_uploading_identical_content_is_a_no_op(api):
    """Idempotency includes cost: the second upload must not re-embed anything."""
    api_module, client = api
    content = b"Parking. Visitor parking is on level two and must be booked a day ahead."
    first = upload(client, "parking.txt", content).json()
    calls_before = api_module._pipeline.embedder.usage.embed_texts
    second = upload(client, "parking.txt", content).json()

    assert second["status"].startswith("skipped")
    assert second["total_chunks"] == first["total_chunks"]
    assert api_module._pipeline.embedder.usage.embed_texts == calls_before


def test_changed_content_replaces_the_old_chunks(api):
    api_module, client = api
    upload(client, "notice.txt", b"Notice. The office closes at six in the evening.")
    chunks_after_first = len(api_module._pipeline.index.chunks)
    body = upload(client, "notice.txt",
                  b"Notice. The office now closes at eight in the evening on weekdays.").json()

    assert body["status"] == "replaced the previous version"
    # Old chunks are gone, so the total did not simply double.
    assert len(api_module._pipeline.index.chunks) <= chunks_after_first + body["chunks"]
    assert "eight" in " ".join(c.text for c in api_module._pipeline.index.chunks.values())


# ------------------------------------------------------------- refusals

def test_an_unsupported_extension_is_415_with_the_allowed_list(api):
    _, client = api
    response = upload(client, "payload.exe", b"MZ\x90\x00binary")
    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]


def test_a_file_over_the_size_limit_is_413(api):
    api_module, client = api
    api_module.settings.max_upload_mb = 0.001          # 1 KB, for the test
    api_module._pipeline.settings.max_upload_mb = 0.001
    response = upload(client, "big.txt", b"x" * 5000)
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_an_empty_file_is_400_not_a_zero_chunk_document(api):
    _, client = api
    assert upload(client, "empty.txt", b"").status_code == 400


def test_the_document_count_is_capped(api):
    api_module, client = api
    api_module._pipeline.settings.max_documents = len(api_module._pipeline.index.docs)
    response = upload(client, "one-too-many.txt", b"Some new and unseen content here.")
    assert response.status_code == 409
    assert "delete one first" in response.json()["detail"]


def test_a_path_in_the_filename_cannot_escape_the_upload_folder(api):
    """Classic traversal: the name is reduced to its last component."""
    api_module, client = api
    body = upload(client, "../../../etc/passwd.txt", b"root. This is not a real passwd file.").json()
    assert body["file"] == "passwd.txt"
    cfg = api_module._pipeline.settings
    assert (cfg.upload_dir / "passwd.txt").exists()      # landed inside the sandbox
    assert not Path("/etc/passwd.txt").exists()          # and nowhere else


def test_uploads_have_their_own_rate_limit(api):
    api_module, client = api
    api_module.settings.uploads_per_hour = 2
    codes = [upload(client, f"doc{i}.txt", f"Unique content number {i} here.".encode()).status_code
             for i in range(3)]
    assert codes[2] == 429


def test_upload_and_delete_require_the_api_key_when_one_is_set(api, monkeypatch):
    api_module, client = api
    from pydantic import SecretStr
    monkeypatch.setattr(api_module.settings, "api_key", SecretStr("secret-123"))
    assert upload(client, "x.txt", b"content here").status_code == 401
    assert client.delete("/documents/refunds.txt").status_code == 401
    # listing stays open: the UI shows it before anyone types a key
    assert client.get("/documents").status_code == 200


# ------------------------------------------------------------- list/delete

def test_documents_lists_what_is_indexed_with_the_limits(api):
    _, client = api
    body = client.get("/documents").json()
    paths = [d["path"] for d in body["documents"]]
    assert "refunds.txt" in paths
    assert body["documents"][0]["chunks"] > 0 and body["documents"][0]["preview"]
    assert body["limits"]["allowed_types"] and body["limits"]["max_documents"] > 0
    assert paths == sorted(paths)


def test_delete_removes_the_chunks_and_changes_the_index_version(api):
    api_module, client = api
    before = client.get("/documents").json()["index_version"]
    body = client.delete("/documents/refunds.txt").json()

    assert body["chunks_removed"] > 0 and body["index_version"] != before
    assert "refunds.txt" not in api_module._pipeline.index.docs
    assert all(c.source != "refunds.txt" for c in api_module._pipeline.index.chunks.values())
    assert client.delete("/documents/refunds.txt").status_code == 404


# --------------------------------------------- uploads stay out of the repo

def test_uploads_are_written_to_the_upload_folder_not_the_seed_corpus(api):
    """A public endpoint must never add files to committed demo material."""
    api_module, client = api
    cfg = api_module._pipeline.settings
    upload(client, "visitor.txt", b"Something a stranger sent to the live demo.")

    assert (cfg.upload_dir / "visitor.txt").exists()
    assert not (cfg.raw_dir / "visitor.txt").exists()


def test_deleting_an_upload_removes_the_file_so_a_restart_cannot_resurrect_it(api):
    api_module, client = api
    cfg = api_module._pipeline.settings
    upload(client, "temporary.txt", b"This should not survive its own deletion.")
    assert (cfg.upload_dir / "temporary.txt").exists()

    body = client.delete("/documents/temporary.txt").json()
    assert body["file_deleted"] is True
    assert not (cfg.upload_dir / "temporary.txt").exists()


def test_deleting_a_seed_document_unindexes_it_but_keeps_the_file(api):
    """refunds.txt is committed demo material: de-indexing must not delete it."""
    api_module, client = api
    cfg = api_module._pipeline.settings
    body = client.delete("/documents/refunds.txt").json()

    assert body["file_deleted"] is False
    assert (cfg.raw_dir / "refunds.txt").exists()


# --------------------------------------------------------------- downloads

def test_an_indexed_document_can_be_downloaded(api):
    """Citations are only checkable if the reader can open the source."""
    _, client = api
    response = client.get("/documents/refunds.txt/download")
    assert response.status_code == 200
    assert b"Refund policy" in response.content


def test_an_uploaded_document_can_be_downloaded_too(api):
    _, client = api
    upload(client, "notes.txt", b"Uploaded content that should come back verbatim.")
    response = client.get("/documents/notes.txt/download")
    assert response.status_code == 200
    assert response.content == b"Uploaded content that should come back verbatim."


def test_download_refuses_a_path_that_escapes_the_document_roots(api):
    _, client = api
    # Not indexed, so it is rejected before the filesystem is ever touched.
    assert client.get("/documents/../../../etc/passwd/download").status_code == 404
    assert client.get("/documents/not-a-real-file.txt/download").status_code == 404


# ------------------------------------------------ a visit leaves no trace

def test_cleanup_removes_every_upload_and_keeps_the_seed_corpus(api):
    """What the browser calls when the tab closes. An interviewer's document
    must not outlive their visit; the demo's own documents must survive it."""
    api_module, client = api
    cfg = api_module._pipeline.settings
    upload(client, "theirs.txt", b"A document belonging to whoever is visiting.")
    assert (cfg.upload_dir / "theirs.txt").exists()

    body = client.post("/documents/cleanup").json()

    assert body["removed"] == ["theirs.txt"]
    assert not (cfg.upload_dir / "theirs.txt").exists()
    assert "refunds.txt" in api_module._pipeline.index.docs
    assert (cfg.raw_dir / "refunds.txt").exists()


def test_cleanup_is_safe_to_call_when_nothing_was_uploaded(api):
    _, client = api
    body = client.post("/documents/cleanup").json()
    assert body["removed"] == []


def test_the_startup_sweep_only_takes_uploads_past_their_ttl(api):
    """sendBeacon is best effort: a crashed tab sends nothing. The sweep is
    the backstop, and it must not take a file that is still in use."""
    import os
    import time as time_module

    api_module, client = api
    cfg = api_module._pipeline.settings
    upload(client, "fresh.txt", b"Uploaded a moment ago, still being used.")
    upload(client, "stale.txt", b"Left behind by a visitor who never came back.")
    old = time_module.time() - 7200
    os.utime(cfg.upload_dir / "stale.txt", (old, old))

    body = api_module._drop_uploads(api_module._pipeline, older_than_s=3600)

    assert body["removed"] == ["stale.txt"]
    assert (cfg.upload_dir / "fresh.txt").exists()


def test_uploads_are_listed_before_seed_documents_and_marked_as_uploads(api):
    api_module, client = api
    api_module._pipeline.settings.eval_corpus_file = "security.txt"
    upload(client, "zz-last-alphabetically.txt", b"An uploaded file that sorts last.")

    documents = client.get("/documents").json()["documents"]
    paths = [d["path"] for d in documents]

    assert paths[0] == "zz-last-alphabetically.txt"   # uploads first
    assert paths[-1] == "security.txt"                # evaluation corpus last
    assert documents[0]["uploaded"] is True
    assert documents[-1]["uploaded"] is False
