FROM python:3.12-slim-bookworm

WORKDIR /app

# sentence-transformers / faiss могут потребовать компилятор на slim-образе
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md MANIFEST.in ./
COPY llm_cache_router ./llm_cache_router
COPY examples/demo ./examples/demo

RUN pip install --no-cache-dir ".[redis,qdrant,fastapi]" "uvicorn[standard]>=0.30"

WORKDIR /app/examples/demo

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
