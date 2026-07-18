from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_cache_router import CacheConfig, LLMRouter, RoutingStrategy
from llm_cache_router.middleware.fastapi import (
    LLMCacheMiddleware,
    add_http_metrics_middleware,
    mount_metrics_endpoint,
)

# Конфигурация демо-приложения из переменных окружения (docker-compose / .env).
_CACHE_BACKEND = os.environ.get("CACHE_BACKEND", "redis")
if _CACHE_BACKEND not in {"redis", "qdrant"}:
    msg = f"Unsupported CACHE_BACKEND={_CACHE_BACKEND!r}; use 'redis' or 'qdrant'"
    raise ValueError(msg)

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not _OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required for the demo stack")

_LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


class ChatRequest(BaseModel):
    message: str


app = FastAPI(
    title="llm-cache-router demo",
    description="FastAPI demo with semantic cache (Redis or Qdrant) and OpenAI",
    version="0.1.0",
)

llm_router = LLMRouter(
    providers={
        "openai": {
            "api_key": _OPENAI_API_KEY,
            "models": [_LLM_MODEL],
        },
    },
    cache=CacheConfig(
        backend=_CACHE_BACKEND,
        threshold=float(os.environ.get("CACHE_THRESHOLD", "0.92")),
        ttl=int(os.environ.get("CACHE_TTL", "3600")),
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
        qdrant_url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
    ),
    strategy=RoutingStrategy.CHEAPEST_FIRST,
    budget={"daily_usd": float(os.environ.get("DAILY_BUDGET_USD", "5.0"))},
)

app.add_middleware(LLMCacheMiddleware, router=llm_router)
add_http_metrics_middleware(app=app)
mount_metrics_endpoint(app=app, router=llm_router, path="/metrics")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "cache_backend": _CACHE_BACKEND}


@app.get("/stats")
async def stats() -> dict:
    return llm_router.stats().model_dump()


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    response = await llm_router.complete(
        messages=[{"role": "user", "content": request.message}],
        model=_LLM_MODEL,
    )
    return response.model_dump()
