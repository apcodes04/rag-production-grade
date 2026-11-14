import json

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def api(monkeypatch, pipeline, tmp_path):
    import rag.api as api_module
    monkeypatch.setattr(api_module, "_pipeline", pipeline)
    monkeypatch.setattr(api_module.settings, "api_rate_limit_per_minute", 3)
    monkeypatch.setattr(api_module.settings, "api_key", None)
    monkeypatch.setattr(api_module.settings, "eval_reports_dir", tmp_path)
    api_module._calls.clear()
    (tmp_path / "baseline-20260101T000000.json").write_text(json.dumps({
        "created_at": "2026-01-01T00:00:00", "config": {"chunk_size": 512},
        "index": {"chunks": 6}, "aggregates": {"recall@5": 1.0, "mrr": 0.95},
        "calibration": {"suggested_min_dense_score": 0.627},
        "results": [{"id": "q01", "question": "x", "error": None}],
    }), encoding="utf-8")
    return api_module, TestClient(api_module.app)


def test_health_reports_the_loaded_index(api):
    _, client = api
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] > 0 and body["index_version"]


def test_ask_returns_an_answer_with_sources_and_a_request_id(api):
    _, client = api
    response = client.post("/ask", json={"question": "How long do refunds take?"})
    body = response.json()
    assert body["refused"] is False and body["citations"] == [1]
    assert body["sources"][0]["source"] == "refunds.txt" and body["timings_ms"]
    assert len(response.headers["X-Request-ID"]) == 12
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_validation_errors_are_422_not_500(api):
    _, client = api
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_rate_limit_kicks_in(api):
    _, client = api
    payload = {"question": "How long do refunds take?"}
    codes = [client.post("/ask", json=payload).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429


def test_api_key_is_enforced_when_configured(api, monkeypatch):
    api_module, client = api
    from pydantic import SecretStr
    monkeypatch.setattr(api_module.settings, "api_key", SecretStr("secret-123"))
    payload = {"question": "How long do refunds take?"}
    assert client.post("/ask", json=payload).status_code == 401
    assert client.post("/ask", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.post("/ask", json=payload, headers={"X-API-Key": "secret-123"})
    assert ok.status_code == 200
    # health stays open on purpose: a load balancer cannot send a secret
    assert client.get("/health").status_code == 200


def test_runs_lists_evaluation_reports_newest_first(api):
    _, client = api
    body = client.get("/runs").json()
    assert len(body["runs"]) == 1
    run = body["runs"][0]
    assert run["aggregates"]["recall@5"] == 1.0 and run["config"]["chunk_size"] == 512


def test_run_detail_returns_every_question(api):
    _, client = api
    run_id = client.get("/runs").json()["runs"][0]["id"]
    detail = client.get(f"/runs/{run_id}").json()
    assert detail["results"][0]["id"] == "q01"
    # The detail carries its own id: the UI titles the page with it.
    assert detail["id"] == run_id
    assert client.get("/runs/does-not-exist").status_code == 404


def test_config_endpoint_pairs_numbers_with_the_settings_that_made_them(api):
    _, client = api
    body = client.get("/config").json()
    assert body["config"]["chunk_size"] and body["index"]["chunks"] > 0


def test_a_provider_outage_becomes_503_not_500(api, monkeypatch):
    api_module, client = api
    from rag.errors import RetryableProviderError

    def boom(*args, **kwargs):
        raise RetryableProviderError("503 UNAVAILABLE")

    monkeypatch.setattr(api_module._pipeline, "answer", boom)
    response = client.post("/ask", json={"question": "anything at all"})
    assert response.status_code == 503
    assert response.json()["error"] == "RetryableProviderError"


def test_cors_allows_every_method_the_app_routes(api):
    """The CORS method list is written by hand and the routes are not.

    This caught a real bug: DELETE /documents shipped while allow_methods was
    still ["GET", "POST"], so the browser preflight failed and the UI reported
    "cannot reach the API" for a server that was running fine. Deriving the
    expected set from the routes means the next new method fails here instead
    of in someone's browser.
    """
    api_module, _ = api
    from fastapi.middleware.cors import CORSMiddleware

    routed = {method for route in api_module.app.routes
              for method in getattr(route, "methods", set())} - {"HEAD", "OPTIONS"}
    allowed = next(mw.kwargs["allow_methods"] for mw in api_module.app.user_middleware
                   if mw.cls is CORSMiddleware)

    assert routed <= set(allowed), f"not allowed through CORS: {routed - set(allowed)}"


def test_a_delete_preflight_is_accepted(api):
    """The exact request the browser makes before a DELETE."""
    _, client = api
    response = client.options("/documents/refunds.txt", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "DELETE",
    })
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]


def test_a_preview_deployment_origin_is_allowed_by_regex(monkeypatch, pipeline, tmp_path):
    """Branch deploys get a fresh subdomain each time, so they cannot be
    enumerated in an exact allowlist. The regex has to cover them, and it has
    to stay scoped to this one project."""
    import importlib

    import rag.api as api_module
    monkeypatch.setenv("CORS_ORIGIN_REGEX", r"https://rag-from-scratch-[a-z0-9-]+\.vercel\.app")
    monkeypatch.setenv("CORS_ORIGINS", "https://rag-from-scratch.vercel.app")
    reloaded = importlib.reload(api_module)   # settings are read at import time
    try:
        monkeypatch.setattr(reloaded, "_pipeline", pipeline)
        client = TestClient(reloaded.app)

        def preflight(origin: str):
            return client.options("/documents/refunds.txt", headers={
                "Origin": origin, "Access-Control-Request-Method": "DELETE"})

        assert preflight("https://rag-from-scratch-git-main-adi.vercel.app").status_code == 200
        assert preflight("https://rag-from-scratch.vercel.app").status_code == 200
        # A different project on the same platform must NOT get through.
        assert "access-control-allow-origin" not in preflight(
            "https://someone-elses-app.vercel.app").headers
    finally:
        # Restore the module for every later test: the CORS middleware is
        # configured at import time, so a reloaded module would leak this
        # test's environment into the rest of the suite.
        monkeypatch.undo()
        importlib.reload(api_module)
