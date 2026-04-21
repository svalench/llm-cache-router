from __future__ import annotations

import pytest

from llm_cache_router import CacheConfig, LLMRouter, RoutingStrategy
from llm_cache_router.models import LLMResponse, LLMStreamChunk


class StubProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, model, temperature=0.0, max_tokens=None):  # noqa: ANN001,ANN201
        del messages, temperature, max_tokens
        self.calls += 1
        return LLMResponse(
            content="ok",
            provider_used="ollama",
            model_used=model,
            input_tokens=100,
            output_tokens=50,
            latency_ms=12,
        )

    async def close(self) -> None:
        return None


class FailingStreamProvider:
    def __init__(self) -> None:
        self.stream_calls = 0

    async def complete(self, messages, model, temperature=0.0, max_tokens=None):  # noqa: ANN001,ANN201
        del messages, model, temperature, max_tokens
        return LLMResponse(content="", provider_used="openai", model_used="gpt-4o")

    async def stream(self, messages, model, temperature=0.0, max_tokens=None):  # noqa: ANN001,ANN201
        del messages, model, temperature, max_tokens
        self.stream_calls += 1
        raise RuntimeError("stream failed")
        yield  # pragma: no cover

    async def close(self) -> None:
        return None


class SuccessfulStreamProvider:
    def __init__(self) -> None:
        self.stream_calls = 0

    async def complete(self, messages, model, temperature=0.0, max_tokens=None):  # noqa: ANN001,ANN201
        del messages, temperature, max_tokens
        return LLMResponse(
            content="ok",
            provider_used="anthropic",
            model_used=model,
            input_tokens=10,
            output_tokens=5,
        )

    async def stream(self, messages, model, temperature=0.0, max_tokens=None):  # noqa: ANN001,ANN201
        del messages, temperature, max_tokens
        self.stream_calls += 1
        yield LLMStreamChunk(delta="hello ", provider_used="anthropic", model_used=model)
        yield LLMStreamChunk(
            delta="world",
            provider_used="anthropic",
            model_used=model,
            is_final=True,
            input_tokens=10,
            output_tokens=5,
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_router_returns_cache_hit_on_second_call() -> None:
    router = LLMRouter(
        providers={"ollama": {"base_url": "http://localhost:11434", "models": ["llama3.2"]}},
        cache=CacheConfig(backend="memory", min_query_length=1, embedding_model="hash"),
        strategy=RoutingStrategy.CHEAPEST_FIRST,
    )
    stub = StubProvider()
    router._providers["ollama"] = stub  # noqa: SLF001

    messages = [{"role": "user", "content": "расскажи про кеш"}]
    first = await router.complete(messages=messages, model="llama3.2")
    second = await router.complete(messages=messages, model="llama3.2")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert stub.calls == 1
    stats = router.stats()
    assert stats.total_requests == 2
    assert stats.cache_hit_rate == 0.5
    assert stats.cache_hits == 1
    assert stats.cache_misses == 1
    assert stats.daily_spend_usd is not None


def test_router_supports_minimax_and_qwen_provider_configs() -> None:
    router = LLMRouter(
        providers={
            "minimax": {"api_key": "minimax-key", "models": ["MiniMax-Text-01"]},
            "qwen": {"api_key": "qwen-key", "models": ["qwen-plus"]},
        },
        cache=CacheConfig(backend="memory", min_query_length=1, embedding_model="hash"),
    )

    assert "minimax" in router._providers  # noqa: SLF001
    assert "qwen" in router._providers  # noqa: SLF001


@pytest.mark.asyncio
async def test_model_usage_stats() -> None:
    router = LLMRouter(
        providers={"openai": {"api_key": "test", "models": ["gpt-4o", "gpt-4o-mini"]}},
        cache=CacheConfig(backend="memory", min_query_length=1, embedding_model="hash"),
    )
    stub = StubProvider()
    router._providers["openai"] = stub  # noqa: SLF001

    messages = [{"role": "user", "content": "model usage stats"}]
    await router.complete(messages=messages, model="gpt-4o")

    stats = router.stats()
    assert "openai/gpt-4o" in stats.model_usage
    stat = stats.model_usage["openai/gpt-4o"]
    assert stat.requests == 1
    assert stat.input_tokens > 0


@pytest.mark.asyncio
async def test_async_context_manager() -> None:
    async with LLMRouter(
        providers={"openai": {"api_key": "test", "models": ["gpt-4o"]}},
    ) as router:
        assert router is not None
    await router.close()


@pytest.mark.asyncio
async def test_router_stream_fallback_chain() -> None:
    router = LLMRouter(
        providers={
            "openai": {"api_key": "test", "models": ["gpt-4o"]},
            "anthropic": {"api_key": "test", "models": ["claude-3-5-sonnet"]},
        },
        cache=CacheConfig(backend="memory", min_query_length=1, embedding_model="hash"),
        strategy=RoutingStrategy.FALLBACK_CHAIN,
        fallback_chain=["openai/gpt-4o", "anthropic/claude-3-5-sonnet"],
    )
    first = FailingStreamProvider()
    second = SuccessfulStreamProvider()
    router._providers["openai"] = first  # noqa: SLF001
    router._providers["anthropic"] = second  # noqa: SLF001

    messages = [{"role": "user", "content": "stream with fallback"}]
    chunks = [chunk async for chunk in router.stream(messages=messages, model="gpt-4o")]

    assert first.stream_calls == 1
    assert second.stream_calls == 1
    assert "".join(chunk.delta for chunk in chunks) == "hello world"
    assert chunks[-1].is_final is True
    assert chunks[-1].provider_used == "anthropic"
    assert chunks[-1].model_used == "claude-3-5-sonnet"

