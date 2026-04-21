from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from llm_cache_router import CacheConfig, LLMRouter, RoutingStrategy
from llm_cache_router.middleware.fastapi import (
    LLMCacheMiddleware,
    add_http_metrics_middleware,
    mount_metrics_endpoint,
)


class ChatRequest(BaseModel):
    message: str


app = FastAPI()

llm_router = LLMRouter(
    providers={
        "openai": {"api_key": "sk-...", "models": ["gpt-4o-mini"]},
        "ollama": {"base_url": "http://localhost:11434", "models": ["llama3.2"]},
    },
    cache=CacheConfig(backend="memory", threshold=0.92, ttl=3600),
    strategy=RoutingStrategy.CHEAPEST_FIRST,
    budget={"daily_usd": 5.0},
)

app.add_middleware(LLMCacheMiddleware, router=llm_router)
add_http_metrics_middleware(app=app)
mount_metrics_endpoint(app=app, router=llm_router, path="/metrics")


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    response = await llm_router.complete(
        messages=[{"role": "user", "content": request.message}],
        model="gpt-4o-mini",
    )
    return response.model_dump()

