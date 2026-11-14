.PHONY: install test lint ingest ask eval smoke api
install:      ## install dependencies
	pip install -r requirements-dev.txt
test:         ## run the full offline test suite (no API key needed)
	python -m pytest
lint:
	ruff check src tests scripts
ingest:       ## build or update the index from data/raw
	python scripts/ingest.py
ask:          ## interactive question loop
	python scripts/ask.py
eval:         ## measure quality against eval/golden_set.json
	python scripts/evaluate.py
smoke:        ## one real call to each Gemini endpoint
	python scripts/smoke_live.py
api:          ## serve the HTTP API
	uvicorn rag.api:app --app-dir src --reload
