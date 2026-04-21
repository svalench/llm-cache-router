from __future__ import annotations

import pytest

from llm_cache_router import (
    CacheConfig,
    LLMRouter,
    add_http_metrics_middleware,
    mount_metrics_endpoint,
)
from llm_cache_router.middleware.fastapi import LLMCacheMiddleware

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")


def test_http_metrics_middleware_exposes_request_metrics() -> None:
    app = fastapi.FastAPI()
    router = LLMRouter(
        providers={"ollama": {"base_url": "http://localhost:11434", "models": ["llama3.2"]}},
        cache=CacheConfig(backend="memory", min_query_length=1, embedding_model="hash"),
    )

    app.add_middleware(LLMCacheMiddleware, router=router)
    add_http_metrics_middleware(app=app)
    mount_metrics_endpoint(app=app, router=router, path="/metrics")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    client = testclient.TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'llm_router_http_requests_total{method="GET",path="/health",status="200"} 1' in metrics.text
    assert "llm_router_http_request_duration_seconds_count" in metrics.text

