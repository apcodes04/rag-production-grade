"""HTTP API: the same pipeline, reachable by a browser.

What a CLI never forces you to think about, and this does:
  one shared pipeline per process (loading the index per request is absurd),
  a request id on every response so a user's complaint is traceable,
  authentication, because this will be on the public internet,
  CORS, because a browser on another origin has to be allowed in explicitly,
  a rate limit, so one client cannot drain the whole API budget,
  error mapping, so a user mistake is 4xx and a provider outage is 503,
  and a health endpoint that reports whether the INDEX is loaded, not just
  whether the process is alive.

Run locally:  uvicorn rag.api:app --app-dir src --reload
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from rag.errors import GuardrailBlocked, RagError, RetryableProviderError, StoreError
from rag.factory import build_pipeline
from rag.ingest import Ingestor
from rag.loader import SUPPORTED_SUFFIXES
from rag.logging_setup import request_id_var, setup_logging
from rag.settings import get_settings
from rag.types import Turn

log = logging.getLogger(__name__)
settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(title=settings.app_name, version="1.0",
              description="Hybrid RAG with guardrails, evaluation and observability")
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origin_list(),
    allow_credentials=False,
    # Every method the app actually routes must be listed. A browser sends a
    # preflight OPTIONS for anything that is not a simple request, and a
    # missing method there fails as a network error with no usable message,
    # so it reads like the server is down when it is running perfectly.
    # test_cors_allows_every_method_the_app_routes keeps this list honest.
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
    # Content-Disposition is not one of the handful of response headers a
    # browser exposes to JS by default, so without this the dashboard could
    # never tell a download from a view in its own fetch calls, only when
    # actually navigating the browser there.
    expose_headers=["Content-Disposition"],
    allow_origin_regex=settings.cors_origin_regex or None)

_pipeline = None
_calls: dict[str, deque[float]] = defaultdict(deque)
_uploads: dict[str, deque[float]] = defaultdict(deque)


def pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline(settings)
    return _pipeline


@app.on_event("startup")
def warm_up() -> None:
    """Load the index and build the pipeline BEFORE the first user request.

    Lazy loading moves a one second cost onto whoever asks the first question,
    which during a demo is always the person you are trying to impress.
    """
    try:
        pipe = pipeline()
        # Anything a previous visitor left behind goes before we serve anyone.
        _drop_uploads(pipe, older_than_s=settings.upload_ttl_minutes * 60)
        stats = pipe.index.stats()
        log.info("warm", extra={"chunks": stats["chunks"], "index": stats["index_version"]})
    except Exception as exc:                 # noqa: BLE001 never block startup
        log.warning("warm up failed, will retry on first request",
                    extra={"error": str(exc)[:200]})


# ----------------------------------------------------------------- plumbing

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    from rag.logging_setup import new_request_id
    rid = new_request_id()
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.0f}"
    log.info("http", extra={"path": request.url.path, "status": response.status_code,
                            "ms": round((time.perf_counter() - started) * 1000, 1)})
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No key configured means open, which is only acceptable on localhost."""
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key.get_secret_value():
        raise HTTPException(401, "invalid or missing X-API-Key header")


def rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.time()
    window = _calls[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.api_rate_limit_per_minute:
        raise HTTPException(429, "rate limit exceeded, try again in a minute")
    window.append(now)


@app.exception_handler(RagError)
async def rag_error_handler(request: Request, exc: RagError) -> JSONResponse:
    status = 503 if isinstance(exc, (RetryableProviderError, StoreError)) else 400
    return JSONResponse(status_code=status,
                        content={"error": type(exc).__name__, "detail": str(exc),
                                 "request_id": request_id_var.get()})


# ------------------------------------------------------------------ schemas

class TurnIn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[TurnIn] = Field(default_factory=list, max_length=20)
    use_cache: bool = True
    # Empty means the whole corpus. Naming documents restricts retrieval to
    # them, so a visitor can ask about one file without the others competing.
    sources: list[str] = Field(default_factory=list, max_length=50)


# ----------------------------------------------------------------- endpoints

@app.get("/health")
def health() -> dict[str, Any]:
    try:
        stats = pipeline().index.stats()
    except StoreError as exc:
        raise HTTPException(503, f"index unavailable: {exc}") from exc
    return {"status": "ok" if stats["chunks"] else "empty_index", **stats}


@app.get("/config", dependencies=[Depends(require_api_key)])
def config() -> dict[str, Any]:
    """What produced the current answers. The dashboard shows this next to the
    metrics, because a number without its configuration is not evidence."""
    return {"config": settings.fingerprint(), "index": pipeline().index.stats(),
            "cache": pipeline().cache.stats() if pipeline().cache else None}


@app.post("/ask", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def ask(body: AskRequest) -> dict[str, Any]:
    try:
        answer = pipeline().answer(
            body.question, [Turn(t.role, t.content) for t in body.history],
            use_cache=body.use_cache, sources=body.sources or None)
    except GuardrailBlocked as exc:
        raise HTTPException(400, str(exc)) from exc
    return answer.to_dict()


def _reports() -> list[Path]:
    directory = Path(settings.eval_reports_dir)
    return sorted(directory.glob("*.json"), reverse=True) if directory.exists() else []


@app.get("/runs")
def runs() -> dict[str, Any]:
    """Every evaluation run, newest first, as summaries for the comparison table."""
    out = []
    for path in _reports():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("unreadable report", extra={"file": path.name})
            continue
        out.append({"id": path.stem, "created_at": report.get("created_at"),
                    "config": report.get("config", {}), "index": report.get("index", {}),
                    "aggregates": report.get("aggregates", {}),
                    "calibration": report.get("calibration", {})})
    return {"runs": out}


@app.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    """One run with every question, for the drill down table."""
    for path in _reports():
        if path.stem == run_id:
            # The id lives in the FILE NAME, not inside the report, so the
            # summary list adds it and the detail has to as well. Without this
            # the UI reads detail.id as undefined.
            return {"id": path.stem, **json.loads(path.read_text(encoding="utf-8"))}
    raise HTTPException(404, f"no evaluation run named {run_id}")


# ------------------------------------------------------------- documents
#
# Uploads use the RUNNING pipeline's embedder and settings, never a freshly
# built one. A second embedder could have a different model or dimension, and
# vectors from two different models in one index silently return nonsense
# instead of failing.

def upload_limit(request: Request) -> None:
    """Uploads cost far more than questions (parsing plus embedding calls), so
    they get their own, much slower bucket."""
    client = request.client.host if request.client else "unknown"
    now = time.time()
    window = _uploads[client]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= settings.uploads_per_hour:
        raise HTTPException(429, f"upload limit is {settings.uploads_per_hour} files per hour")
    window.append(now)


def _doc_view(index, path: str) -> dict[str, Any]:
    record = index.docs[path]
    chunks = [index.chunks[cid] for cid in record.chunk_ids if cid in index.chunks]
    return {
        "path": path, "doc_id": record.doc_id, "size_bytes": record.size,
        "chunks": len(chunks),
        "pages": len({c.page for c in chunks if c.page is not None}),
        "tokens": sum(c.token_count for c in chunks),
        "preview": chunks[0].text[:180] if chunks else "",
    }


def _document_order(pipe, path: str) -> tuple[int, str]:
    """Uploads first, then the seed corpus, with the evaluation document last.

    The evaluation corpus is the least interesting thing a visitor can ask
    about, and it is the one already measured on the Evaluation page, so it
    goes to the bottom rather than greeting everyone at the top.
    """
    if path == pipe.settings.eval_corpus_file:
        return (2, path)
    is_upload = (pipe.settings.upload_dir / path).exists()
    return (0 if is_upload else 1, path)


@app.get("/documents")
def documents() -> dict[str, Any]:
    pipe = pipeline()
    index = pipe.index
    ordered = sorted(index.docs, key=lambda path: _document_order(pipe, path))
    return {
        "documents": [{**_doc_view(index, path),
                       "uploaded": (pipe.settings.upload_dir / path).exists()}
                      for path in ordered],
        "total_chunks": len(index.chunks),
        "index_version": index.index_version(),
        "limits": {
            "max_upload_mb": pipe.settings.max_upload_mb,
            "max_documents": pipe.settings.max_documents,
            "allowed_types": sorted(SUPPORTED_SUFFIXES),
        },
    }


@app.post("/documents", dependencies=[Depends(require_api_key), Depends(upload_limit)])
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Take one document, index it, and report exactly what happened to it.

    Every check below exists because this endpoint is public on the demo. An
    unknown extension, a 2 GB file and a thousand documents are all things a
    stranger can send, and none of them may take the service down.
    """
    pipe = pipeline()
    cfg = pipe.settings
    name = Path(file.filename or "upload").name       # drop any directory parts
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(415, f"{suffix or 'that file type'} is not supported. "
                                 f"Allowed: {', '.join(sorted(SUPPORTED_SUFFIXES))}")

    payload = await file.read()
    size_mb = len(payload) / (1024 * 1024)
    if not payload:
        raise HTTPException(400, "that file is empty")
    if size_mb > cfg.max_upload_mb:
        raise HTTPException(413, f"file is {size_mb:.1f} MB, the limit is {cfg.max_upload_mb} MB")

    index = pipe.index
    if name not in index.docs and len(index.docs) >= cfg.max_documents:
        raise HTTPException(409, f"this demo holds at most {cfg.max_documents} documents, "
                                 f"delete one first")

    # Visitor uploads go to the upload folder, never to the committed seed
    # corpus: a public endpoint must not be able to add files to the repo.
    cfg.upload_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.upload_dir / name
    target.write_bytes(payload)

    started = time.perf_counter()
    report = Ingestor(index, pipe.embedder, cfg).ingest([target])
    ingest_ms = round((time.perf_counter() - started) * 1000, 1)

    if report.failed:
        raise HTTPException(422, f"could not read {name}: {report.failed[0][1]}")

    status = ("skipped (identical content already indexed)" if report.skipped else
              "duplicate of another document" if report.duplicates else
              "replaced the previous version" if report.updated else "added")
    view = _doc_view(index, name) if name in index.docs else {"chunks": 0, "tokens": 0, "pages": 0}
    chunk_ids = index.docs[name].chunk_ids if name in index.docs else []
    log.info("document ingested", extra={"file": name, "chunks": view["chunks"], "ms": ingest_ms})
    return {
        "file": name, "size_mb": round(size_mb, 3), "status": status,
        "chunks": view["chunks"], "tokens": view["tokens"], "pages": view["pages"],
        "ingest_ms": ingest_ms,
        "index_version": index.index_version(),
        "total_chunks": len(index.chunks),
        "report": report.summary(),
        "sample_chunks": [
            {"chunk_id": cid, "page": index.chunks[cid].page,
             "tokens": index.chunks[cid].token_count, "text": index.chunks[cid].text[:400]}
            for cid in chunk_ids[:3] if cid in index.chunks
        ],
    }


def _resolve_document(cfg, path: str) -> Path:
    """Map an indexed document key back to a file on disk, safely.

    The key comes from the URL, so it is attacker controlled. Resolving it and
    then checking containment is what stops "../../etc/passwd" and symlinks
    from escaping the document roots: checking the string before resolving it
    would miss both.
    """
    for root in cfg.document_roots():
        candidate = (root / path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue                       # escaped this root, try the next
        if candidate.is_file():
            return candidate
    raise HTTPException(404, f"no stored file for {path}")


@app.get("/documents/{path:path}/download")
def download_document(path: str) -> FileResponse:
    """Serve the original file an answer was built from, forced to save
    rather than open.

    Citations are only checkable if the reader can open the source, so the
    demo lets anyone download the documents it is answering from. The
    dashboard and this API run on different ports in dev, and a browser
    never honours an <a download> hint across origins, so without an
    explicit header here the browser just navigates to the file (opening a
    PDF inline) instead of saving it. The header is what actually forces
    the save, on any origin.
    """
    pipe = pipeline()
    if path not in pipe.index.docs:
        raise HTTPException(404, f"no indexed document named {path}")
    resolved = _resolve_document(pipe.settings, path)
    return FileResponse(
        resolved, filename=resolved.name,
        headers={"Content-Disposition": f'attachment; filename="{resolved.name}"'},
    )


@app.get("/documents/{path:path}/view")
def view_document(path: str) -> FileResponse:
    """Serve the same file for viewing in the browser instead of saving it.

    A visitor comparing an answer against its source wants to read the
    document, not have it dropped in their downloads folder. Explicit
    "inline" is what tells the browser to render it (a PDF opens in the
    built-in viewer) rather than defaulting to a save, which is what some
    browsers do for an unrecognised disposition.
    """
    pipe = pipeline()
    if path not in pipe.index.docs:
        raise HTTPException(404, f"no indexed document named {path}")
    resolved = _resolve_document(pipe.settings, path)
    return FileResponse(
        resolved, filename=resolved.name,
        headers={"Content-Disposition": f'inline; filename="{resolved.name}"'},
    )


@app.delete("/documents/{path:path}", dependencies=[Depends(require_api_key)])
def remove_document(path: str) -> dict[str, Any]:
    """Drop a document from the index, and delete the file if it was uploaded.

    Removing it from the index alone is not enough: the file would still be on
    disk, and the next restart re-ingests the whole upload folder and brings it
    straight back. Seed corpus files are never deleted, because those are the
    sample document every visitor gets, not something a visitor added: a
    visitor can download it, never remove it, so there is always at least one
    document to demo against.
    """
    pipe = pipeline()
    cfg, index = pipe.settings, pipe.index
    if path not in index.docs:
        raise HTTPException(404, f"no indexed document named {path}")

    upload_root = cfg.upload_dir.resolve()
    candidate = (upload_root / path).resolve()
    is_upload = candidate.is_file() and candidate.is_relative_to(upload_root)
    if not is_upload:
        raise HTTPException(403, f"{path} is the seed sample document and cannot be deleted")

    removed = index.remove_doc(path)
    index.save(cfg.index_dir)

    file_deleted = False
    try:
        candidate.unlink()
        file_deleted = True
    except OSError as exc:              # read only disk: de-indexing still stands
        log.warning("could not delete upload",
                    extra={"file": path, "error": str(exc)[:200]})

    log.info("document removed", extra={"file": path, "chunks_removed": removed,
                                        "file_deleted": file_deleted})
    return {"removed": path, "chunks_removed": removed, "file_deleted": file_deleted,
            "total_chunks": len(index.chunks), "index_version": index.index_version()}


@app.post("/documents/cleanup")
def cleanup_uploads() -> dict[str, Any]:
    """Remove every uploaded document. Seed documents are never touched.

    The browser calls this with sendBeacon when the tab closes, so an
    interviewer's file does not outlive their visit. sendBeacon can only POST
    and cannot be relied on (a crashed tab sends nothing), which is why
    sweep_old_uploads below runs as well.
    """
    return _drop_uploads(pipeline(), older_than_s=None)


def _drop_uploads(pipe, *, older_than_s: float | None) -> dict[str, Any]:
    cfg, index = pipe.settings, pipe.index
    upload_root = cfg.upload_dir.resolve()
    now = time.time()
    removed, chunks = [], 0

    for path in list(index.docs):
        file = (upload_root / path)
        if not file.is_file():
            continue                       # seed corpus, not ours to delete
        if older_than_s is not None and now - file.stat().st_mtime < older_than_s:
            continue
        chunks += index.remove_doc(path)
        try:
            file.unlink()
        except OSError as exc:             # read only disk: de-indexing stands
            log.warning("could not delete upload",
                        extra={"file": path, "error": str(exc)[:200]})
        removed.append(path)

    if removed:
        index.save(cfg.index_dir)
        log.info("uploads cleared", extra={"files": len(removed), "chunks": chunks})
    return {"removed": removed, "chunks_removed": chunks,
            "total_chunks": len(index.chunks), "index_version": index.index_version()}
