# Build context is the project root (see docker-compose.yml's `context: ..`).
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependencies first, in their own layer, so code-only changes don't bust
# the (slow: torch, sentence-transformers) dependency install cache.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY config/ ./config/
COPY src/ ./src/
RUN uv sync --frozen

# data/ (chunks, the persisted Chroma/BM25 indexes) is mounted as a volume
# at runtime, not baked into the image -- see docker-compose.yml. It's the
# output of Module 2/3's pipelines, run on the host or in this container.

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
