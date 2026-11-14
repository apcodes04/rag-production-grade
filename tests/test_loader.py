from rag.loader import normalize, strip_repeated_edges, unwrap_lines


def test_hyphen_break_and_ligature():
    assert normalize("exam-\nple of ﬁnance") == "example of finance"


def test_unwrap_joins_wrapped_lines_but_keeps_lists_and_paragraphs():
    text = ("This is a long wrapped line that keeps going and going on\nand then ends here.\n\n"
            "- first item that is fairly long and keeps going on and on and on\n- second item")
    out = unwrap_lines(text)
    assert "going on and then ends here." in out
    assert "\n- second item" in out
    assert "\n\n" in out


def test_repeated_header_and_page_number_removed():
    topics = ["apples", "rivers", "engines", "poetry", "volcanoes"]
    pages = [
        f"ACME Confidential Report\nIntro line about {t}.\nSecond line discussing {t} in depth.\n"
        f"Third line with more on {t}.\nFinal remarks regarding {t}.\nPage {i} of 5"
        for i, t in enumerate(topics, start=1)
    ]
    cleaned = strip_repeated_edges(pages)
    assert all("ACME Confidential" not in p and "of 5" not in p for p in cleaned)
    assert all(f"Intro line about {t}" in cleaned[i] and f"Final remarks regarding {t}" in cleaned[i]
               for i, t in enumerate(topics))


def test_short_documents_are_left_alone():
    pages = ["Header\nbody one\nFooter", "Header\nbody two\nFooter"]
    assert strip_repeated_edges(pages) == pages
