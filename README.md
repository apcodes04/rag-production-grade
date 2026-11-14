---
title: Production RAG From Scratch
emoji: 🔎
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Hybrid retrieval RAG with guardrails, no framework underneath
---

# rag-from-scratch

A production grade hybrid RAG system, written module by module, with the
failure modes handled rather than ignored. Built as the practical half of the
RAG study in `../notes`; the reference project it improves on is in
`../code/ProductionRAG_hybrid_retrieval_bm25_CLI-main` (read only) and its
audit is `../notes/01-codebase-audit.md`.

```
PDF / TXT / MD / DOCX / CSV
   |
   v
loader.py      clean text, strip running headers, stable doc_id per file
   |
   v
chunker.py     recursive, token budgeted, structure aware chunks
   |
   +--> embeddings (Gemini, batched + retried) --> vector_index  (dense)
   +--> tokenizer -------------------------------> bm25_index    (sparse)
                       both saved together as one versioned index
   |
   v
query  ->  input guard  ->  rewrite follow ups  ->  cache lookup
       ->  hybrid retrieval (dense + sparse, RRF)
       ->  relevance gate (refuse here if nothing is relevant)
       ->  rerank  ->  prompt  ->  LLM
       ->  output guard (citations real? answer grounded?)  ->  cache store
       ->  Answer (text, sources, scores, timings, usage, request id)
```

## Quick start

```bash
pip install -r requirements-dev.txt
cp .env.example .env          # then paste your Gemini key into .env

python -m pytest              # 120 tests, offline, no API key needed
python scripts/list_models.py # which models your key can call
python scripts/smoke_live.py  # one real call per Gemini endpoint
python scripts/ingest.py      # build the index from data/raw
python scripts/ask.py "What are the three types of AI?"
python scripts/ask.py         # interactive, with follow up questions
python scripts/evaluate.py    # measure quality against the golden set
```

Run everything offline with no API key by setting `PROVIDER=fake` in `.env`.

## What is in here

| Path | What it does |
|---|---|
| `src/rag/settings.py` | every tunable value, typed and validated at startup |
| `src/rag/logging_setup.py` | one JSON log line per event, with a request id |
| `src/rag/loader.py` | documents to clean, page tagged text |
| `src/rag/chunker.py` | text to token budgeted chunks |
| `src/rag/tokenizer.py` | text to terms for BM25 |
| `src/rag/providers/` | Gemini and fake embedding / LLM providers |
| `src/rag/stores/` | BM25 index, vector index, and the versioned index bundle |
| `src/rag/retrieval/` | fusion, hybrid search, reranking, query rewriting |
| `src/rag/guardrails/` | input checks and output checks |
| `src/rag/prompts.py` | every prompt, versioned |
| `src/rag/cache.py` | answer cache keyed on everything that changes the answer |
| `src/rag/ingest.py` | incremental ingestion with a manifest |
| `src/rag/pipeline.py` | the orchestrator |
| `src/rag/api.py` | FastAPI service |
| `src/rag/evaluation/` | metrics and the evaluation harness |
| `eval/golden_set.json` | 13 labelled questions, 2 of them deliberately out of scope |

`ARCHITECTURE.md` explains why each file exists, why each file format was
chosen, and what is deliberately left out.


## Running it

```bash
pip install -r requirements.txt
cp .env.example .env            # add your GEMINI_API_KEY
python scripts/ingest.py        # build the index from data/raw/
python scripts/ask.py "your question"
uvicorn rag.api:app --app-dir src --reload
```

The UI is a separate Vite app:

```bash
cd dashboard
npm install
npm run dev                     # http://localhost:5173
```

## Deploying

The backend is one container. It reads `PORT` from the environment, seeds its
index at boot if the disk is empty, and takes every secret as an environment
variable, so the same image runs locally and on any container host.

```bash
docker build -t rag-from-scratch .
docker run -p 7860:7860 -e GEMINI_API_KEY=... rag-from-scratch
```

On Hugging Face Spaces, pick the Docker SDK and set these as Space secrets:

| Secret | Why |
|---|---|
| `GEMINI_API_KEY` | embeddings and generation |
| `API_KEY` | required on `/ask`, `/documents` upload and delete |
| `CORS_ORIGINS` | the frontend origin, comma separated |
| `CORS_ORIGIN_REGEX` | optional, to also allow branch preview deployments |

The frontend deploys as a static site. Set `VITE_API_URL` to the Space URL and
`VITE_API_KEY` to the same value as `API_KEY`.

A frontend API key is visible to anyone who opens the network tab. It is a
throttle on casual abuse of the demo, not a secret: the real protections are
the per IP rate limits, the upload size and count caps, and the fact that the
Gemini key never leaves the server.
