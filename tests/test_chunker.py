from rag.chunker import Chunk, chunk_pages, count_tokens, split_text
from rag.loader import Page

SIZE, OVERLAP = 100, 10


def test_giant_sentence_respects_budget():           # reference bug B1
    text = " ".join(f"w{i}" for i in range(2000))     # no punctuation at all
    chunks = split_text(text, SIZE, OVERLAP)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= SIZE for c in chunks)


def test_huge_blob_without_spaces_is_sliced():
    chunks = split_text("A" * 20000, SIZE, OVERLAP)
    assert all(count_tokens(c) <= SIZE for c in chunks)
    assert "".join(chunks) == "A" * 20000


def test_short_section_is_kept():                     # reference bug B2
    chunks = split_text("Refund Policy. 30 days only.", 512, 64)
    assert chunks == ["Refund Policy. 30 days only."]


def test_no_content_lost():
    sentences = [f"Sentence number {i} talks about topic{i}." for i in range(200)]
    text = " ".join(sentences)
    chunks = split_text(text, SIZE, OVERLAP)
    seen = " ".join(chunks)
    assert all(f"topic{i}" in seen for i in range(200))
    assert all(count_tokens(c) <= SIZE for c in chunks)


def test_overlap_repeats_whole_sentences():
    text = " ".join(f"Sentence number {i} talks about topic{i}." for i in range(200))
    chunks = split_text(text, SIZE, 20)
    for a, b in zip(chunks, chunks[1:], strict=False):
        last_sentence = a.split(". ")[-1].rstrip(".")
        assert last_sentence in b                      # overlap carried, not cut mid sentence


def test_heading_stays_with_following_text():
    para1 = " ".join(["alpha"] * 80) + "."
    para2 = " ".join(["beta"] * 80) + "."
    text = f"{para1}\n\nPricing Plans\n\n{para2}"
    chunks = split_text(text, SIZE, OVERLAP)
    assert not any(c.rstrip().endswith("Pricing Plans") for c in chunks)
    assert any(c.startswith("Pricing Plans") for c in chunks)


def test_ids_unique_deterministic_and_doc_scoped():   # reference bug B6
    same = "Terms and conditions apply to every order placed with us."
    pages = [Page("docA", "a.txt", None, same), Page("docB", "b.txt", None, same)]
    first = chunk_pages(pages, SIZE, OVERLAP)
    again = chunk_pages(pages, SIZE, OVERLAP)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in again]
    assert len({c.chunk_id for c in first}) == 2
    assert isinstance(first[0], Chunk) and first[0].source == "a.txt"
