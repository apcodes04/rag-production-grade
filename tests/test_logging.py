"""Grader for Lesson 2 (logging). Run: python -m pytest tests/test_logging.py -q"""
import json
import logging

from rag.logging_setup import new_request_id, setup_logging


def _lines(capsys):
    return [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]


def test_log_line_is_json_with_core_fields(capsys):
    setup_logging("INFO")
    logging.getLogger("t").info("hello")
    (rec,) = _lines(capsys)
    assert rec["msg"] == "hello" and rec["level"] == "INFO" and rec["logger"] == "t"
    assert "ts" in rec and rec["request_id"] == "-"


def test_extra_fields_are_included(capsys):
    setup_logging("INFO")
    logging.getLogger("t").info("done", extra={"chunks": 6, "source": "a.pdf"})
    (rec,) = _lines(capsys)
    assert rec["chunks"] == 6 and rec["source"] == "a.pdf"


def test_request_id_is_attached(capsys):
    setup_logging("INFO")
    rid = new_request_id()
    logging.getLogger("t").info("x")
    (rec,) = _lines(capsys)
    assert len(rid) == 12 and rec["request_id"] == rid


def test_level_filters_and_no_duplicate_handlers(capsys):
    setup_logging("WARNING")
    setup_logging("WARNING")               # calling twice must not double print
    log = logging.getLogger("t")
    log.info("hidden")
    log.warning("shown")
    recs = _lines(capsys)
    assert [r["msg"] for r in recs] == ["shown"]
