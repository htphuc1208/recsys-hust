FROM python:3.10-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
RUN python -m pip wheel --wheel-dir /wheels --extra-index-url https://download.pytorch.org/whl/cpu ".[serving]"


FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVING_CONFIG=/app/configs/serving/mind_production.yaml \
    MODEL_BUNDLE_DIR=/app/artifacts/serving/current \
    OMP_NUM_THREADS=1

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels "recsys-hust[serving]" \
    && rm -rf /wheels \
    && useradd --create-home --uid 10001 recsys \
    && mkdir -p /app/artifacts/serving \
    && chown -R recsys:recsys /app

COPY --chown=recsys:recsys configs/ configs/

USER recsys

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
