"""The .docx and .csv readers. Both exist because an interviewer uploading
their own file is far more likely to have a Word document or a spreadsheet
export than a markdown file."""
from __future__ import annotations

import csv

import pytest

from rag.loader import SUPPORTED_SUFFIXES, load_document


def test_csv_rows_become_readable_records_not_comma_soup(tmp_path):
    path = tmp_path / "tickets.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticket", "status", "owner"])
        writer.writerow(["T-1", "resolved", "priya"])
        writer.writerow(["T-2", "open", "sam"])

    text = "\n".join(page.text for page in load_document(path))
    # The column name travels with the value, which is what makes it retrievable.
    assert "status: resolved" in text and "owner: priya" in text
    assert "T-2" in text


def test_csv_rows_are_batched_into_pages_so_one_file_is_not_one_chunk(tmp_path):
    path = tmp_path / "big.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "note"])
        for i in range(95):
            writer.writerow([i, f"note number {i}"])

    pages = load_document(path)
    assert len(pages) == 3                      # 40 + 40 + 15 rows
    assert [p.page for p in pages] == [1, 2, 3]


def test_an_empty_csv_yields_no_pages(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert load_document(path) == []


def test_csv_tolerates_short_rows(tmp_path):
    path = tmp_path / "ragged.csv"
    path.write_text("a,b,c\n1,,3\n2\n", encoding="utf-8")
    text = "\n".join(page.text for page in load_document(path))
    assert "a: 1" in text and "c: 3" in text
    assert "b:" not in text                     # blank cells are dropped, not padded


def test_docx_reads_paragraphs_headings_and_table_cells(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Leave Policy", level=1)
    document.add_paragraph("Employees receive twenty five days of paid leave.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Band"
    table.cell(0, 1).text = "Extra days"
    table.cell(1, 0).text = "Senior"
    table.cell(1, 1).text = "Five"
    path = tmp_path / "policy.docx"
    document.save(str(path))

    pages = load_document(path)
    text = pages[0].text
    assert "## Leave Policy" in text                  # heading marked for the chunker
    assert "twenty five days" in text
    assert "Senior | Five" in text                    # table content kept, not dropped


def test_both_new_types_are_in_the_supported_set():
    assert {".docx", ".csv"} <= SUPPORTED_SUFFIXES
