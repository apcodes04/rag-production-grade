from rag.tokenizer import STOPWORDS, tokenize


def test_punctuation_is_stripped():
    # The reference tokenizer was text.lower().split(), so "simple." never
    # matched the query "simple". This is the fix.
    assert "simple" in tokenize("Our policy is simple.")


def test_stopwords_removed():
    assert tokenize("the cat and the dog") == ["cat", "dog"]
    assert "the" in STOPWORDS


def test_plural_folding_unifies_refund_and_refunds():
    assert tokenize("refunds") == tokenize("refund")
    assert tokenize("policies") == ["policy"]
    assert tokenize("boxes") == ["box"]


def test_short_and_irregular_words_are_left_alone():
    assert tokenize("gas bus status") == ["gas", "bus", "status"]


def test_unicode_is_normalised():
    assert tokenize("ﬁnance naïve") == ["finance", "naïve"] or tokenize("ﬁnance")[0] == "finance"


def test_numbers_and_codes_survive():
    assert tokenize("error E4012 on 2026-01-05") == ["error", "e4012", "2026", "01", "05"]
