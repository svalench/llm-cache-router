from __future__ import annotations

import pytest

from llm_cache_router import CacheConfig, LLMRouter, RoutingStrategy, build_prometheus_metrics
from llm_cache_router.observability.prometheus import HTTPMetricsCollector
from llm_cache_router.models import LLMResponse


class StubProvider:
    async def complete(self, messages, model, temperature=0.0, max_tokens=None):  # noqa: ANN001,ANN201
        del messages, temperature, max_tokens
        return LLMResponse(
            content="ok",
            provider_used="ollama",
            model_used=model,
            input_tokens=100,
            output_tokens=10,
            latency_ms=10,
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_prometheus_payload_contains_key_metrics() -> None:
    router = LLMRouter(
        providers={"ollama": {"base_url": "http://localhost:11434", "models": ["llama3.2"]}},
        cache=CacheConfig(backend="memory", min_query_length=1, embedding_model="hash"),
        strategy=RoutingStrategy.CHEAPEST_FIRST,
        budget={"daily_usd": 2.0},
    )
    router._providers["ollama"] = StubProvider()  # noqa: SLF001

    messages = [{"role": "user", "content": "покажи метрики"}]
    await router.complete(messages=messages, model="llama3.2")
    await router.complete(messages=messages, model="llama3.2")

    payload = build_prometheus_metrics(router)
    assert "llm_router_requests_total 2.0" in payload
    assert "llm_router_cache_hits_total 1.0" in payload
    assert "llm_router_cache_misses_total 1.0" in payload
    assert 'llm_router_provider_requests_total{provider="ollama"} 1' in payload
    assert "llm_router_daily_spend_usd" in payload


def test_http_metrics_collector_renders_prometheus_histogram() -> None:
    collector = HTTPMetricsCollector(buckets=(0.1, 0.5, 1.0))
    collector.observe(method="GET", path="/health", status_code=200, duration_seconds=0.2)
    collector.observe(method="GET", path="/health", status_code=200, duration_seconds=0.7)

    payload = collector.build_metrics()
    assert 'llm_router_http_requests_total{method="GET",path="/health",status="200"} 2' in payload
    assert (
        'llm_router_http_request_duration_seconds_bucket{method="GET",path="/health",le="0.5"} 1'
        in payload
    )
    assert (
        'llm_router_http_request_duration_seconds_bucket{method="GET",path="/health",le="+Inf"} 2'
        in payload
    )

