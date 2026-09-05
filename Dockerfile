# ReservoirX-D
# Single worker by default: the in-memory matrix store is per-process, so
# matrix_ids issued by one worker are invisible to the others. Scale out only
# after switching RXD_STORE_BACKEND to redis.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8080}/health || exit 1

# Shell form on purpose: hosts like Render and Cloud Run assign the port through
# $PORT at runtime, and exec form would pass the literal string "$PORT" to uvicorn.
# Falls back to 8080 so local `docker run -p 8080:8080` still works unchanged.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
