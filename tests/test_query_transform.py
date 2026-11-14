from rag.providers.fake import FakeLLM
from rag.retrieval.query_transform import condense_question, needs_rewrite
from rag.types import Turn

HISTORY = [Turn("user", "Tell me about express shipping"),
           Turn("assistant", "Express shipping delivers next day.")]


def test_no_history_means_no_rewrite():
    assert not needs_rewrite("What is express shipping?", [])


def test_pronouns_and_fragments_trigger_a_rewrite():
    assert needs_rewrite("what about its cost?", HISTORY)
    assert needs_rewrite("and refunds?", HISTORY)
    assert not needs_rewrite("What are the password rules for admin accounts?", HISTORY)


def test_condense_uses_the_llm_and_returns_one_line():
    llm = FakeLLM(scripted=["What is the cost of express shipping?"])
    assert condense_question(llm, "what about its cost?", HISTORY) == \
        "What is the cost of express shipping?"
    assert "rewrite" in llm.calls[0][0].lower()


def test_a_rambling_rewrite_is_rejected_in_favour_of_the_original():
    llm = FakeLLM(scripted=["x" * 500])
    assert condense_question(llm, "what about its cost?", HISTORY) == "what about its cost?"


def test_an_empty_rewrite_is_rejected():
    assert condense_question(FakeLLM(scripted=["  "]), "and cost?", HISTORY) == "and cost?"
