# One image for local Docker and for Hugging Face Spaces.
#
# Spaces differences this handles:
#   - the platform routes to port 7860, so PORT is read from the environment
#   - the container runs as a non root user whose home must be writable
#   - the disk starts empty, so the index is seeded at boot (bootstrap_index.py)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=7860

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY eval/ ./eval/
COPY data/raw/ ./data/raw/

# Spaces runs as uid 1000. Everything the app writes (index, cache, uploads)
# must be writable by that user, and the API key arrives as a secret at run
# time: it is never baked into the image.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/index /app/data/cache /app/data/raw \
    && chown -R app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import os,urllib.request;urllib.request.urlopen(f\"http://localhost:{os.environ['PORT']}/health\")"

# Seed the index, then serve. If seeding fails the container still starts, so a
# bad document can never take the whole service down.
CMD ["sh", "-c", "python scripts/bootstrap_index.py || echo 'seeding failed, serving anyway'; exec uvicorn rag.api:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
