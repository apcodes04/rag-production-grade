from pathlib import Path

from rag.ingest import Ingestor
from rag.stores.index import Index


def run(index, embedder, settings, **kwargs):
    return Ingestor(index, embedder, settings).ingest_dir(**kwargs)


def test_first_run_indexes_everything(settings, embedder):
    index = Index(embedder.dim, embedder.name)
    report = run(index, embedder, settings)
    assert len(report.added) == 3 and report.chunks_added > 0
    assert len(index.chunks) == report.chunks_added
    assert len(index.vectors) == len(index.chunks) == len(index.bm25)


def test_second_run_costs_nothing(settings, embedder, index):
    calls_before = embedder.usage.embed_calls
    report = run(index, embedder, settings)
    assert len(report.skipped) == 3 and report.chunks_added == 0
    assert embedder.usage.embed_calls == calls_before      # zero API calls


def test_a_new_file_is_picked_up(settings, embedder, index):
    # The reference project loaded the existing index and silently ignored
    # new files (audit bug B5). This test is that bug, frozen.
    (settings.raw_dir / "returns.txt").write_text(
        "Return label. Print the prepaid return label and attach it to the parcel.",
        encoding="utf-8")
    report = run(index, embedder, settings)
    assert report.added == ["returns.txt"] and report.chunks_added > 0
    assert any("prepaid return label" in c.text for c in index.chunks.values())


def test_a_changed_file_replaces_its_old_chunks(settings, embedder, index):
    (settings.raw_dir / "refunds.txt").write_text(
        "Refund policy. Refunds are now issued within 14 days of purchase.", encoding="utf-8")
    report = run(index, embedder, settings)
    assert report.updated == ["refunds.txt"] and report.chunks_removed > 0
    texts = " ".join(c.text for c in index.chunks.values())
    assert "14 days" in texts and "30 days" not in texts


def test_a_deleted_file_is_pruned(settings, embedder, index):
    (settings.raw_dir / "shipping.txt").unlink()
    report = run(index, embedder, settings, remove_missing=True)
    assert report.removed == ["shipping.txt"]
    assert not any(c.source == "shipping.txt" for c in index.chunks.values())


def test_the_same_content_under_a_new_name_is_not_indexed_twice(settings, embedder, index):
    copy = settings.raw_dir / "refunds_copy.txt"
    copy.write_text((settings.raw_dir / "refunds.txt").read_text(encoding="utf-8"), encoding="utf-8")
    report = run(index, embedder, settings)
    assert report.duplicates == ["refunds_copy.txt"] and report.chunks_added == 0


def test_one_broken_file_does_not_stop_the_run(settings, embedder):
    (settings.raw_dir / "broken.pdf").write_bytes(b"this is not a pdf at all")
    index = Index(embedder.dim, embedder.name)
    report = run(index, embedder, settings)
    assert len(report.failed) == 1 and report.failed[0][0] == "broken.pdf"
    assert len(report.added) == 3 and index.chunks


def test_unsupported_types_are_reported_not_guessed(settings, embedder):
    (settings.raw_dir / "notes.docx").write_bytes(b"PK\x03\x04 fake docx")
    index = Index(embedder.dim, embedder.name)
    report = Ingestor(index, embedder, settings).ingest(
        [settings.raw_dir / "notes.docx"])
    assert report.failed and "unsupported" in report.failed[0][1]


def test_the_index_is_saved_and_reloadable(settings, embedder, index):
    loaded = Index.load(settings.index_dir, dim=embedder.dim, embedding_model=embedder.name)
    assert loaded.index_version() == index.index_version()
    assert len(loaded.docs) == 3
    assert (Path(settings.index_dir) / "current.json").exists()


def versions(settings) -> list[str]:
    return sorted(p.name for p in settings.index_dir.glob("v*") if p.is_dir())


def test_a_no_op_run_does_not_write_a_new_index_version(settings, embedder, index):
    before = versions(settings)
    run(index, embedder, settings)
    assert versions(settings) == before      # nothing changed, so nothing was rewritten


def test_a_real_change_does_write_a_new_version(settings, embedder, index):
    before = versions(settings)
    (settings.raw_dir / "new.txt").write_text("Gift cards expire after one year.", encoding="utf-8")
    run(index, embedder, settings)
    assert len(versions(settings)) == len(before) + 1


def test_a_renamed_file_survives_a_prune(index, embedder, settings):
    """A rename is a delete plus an add, and order decides whether it works.

    Pruning after indexing meant the new name hit the duplicate check against
    the old entry (same bytes, same hash), got skipped, and then the old entry
    was removed as missing. The document vanished from a run that reported
    no failures.
    """
    original = settings.raw_dir / "refunds.txt"
    renamed = settings.raw_dir / "refund-policy.txt"
    original.rename(renamed)

    report = Ingestor(index, embedder, settings).ingest_dir(remove_missing=True)

    assert "refund-policy.txt" in index.docs
    assert "refunds.txt" not in index.docs
    assert "refund-policy.txt" in report.added
    assert "refunds.txt" in report.removed
    assert index.docs["refund-policy.txt"].chunk_ids
    # The content is still retrievable, which is the thing that actually broke.
    assert any("refund" in c.text.lower() for c in index.chunks.values())
