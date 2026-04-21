from __future__ import annotations

import pytest

from llm_cache_router import CacheConfig, LLMRouter
from llm_cache_router.models import LLMResponse, WarmupEntry


class WarmupStubProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, model, temperature=0.0, max_tokens=None):  # noqa: ANN001,ANN201
        del messages, temperature, max_tokens
        self.calls += 1
        return LLMResponse(
            content="warmup-response",
            provider_used="openai",
            model_used=model,
            input_tokens=5,
            output_tokens=2,
            latency_ms=3,
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_warmup_skips_cached() -> None:
    router = LLMRouter(
        providers={"openai": {"api_key": "test", "models": ["gpt-4o-mini"]}},
        cache=CacheConfig(backend="memory", min_query_length=1, embedding_model="hash"),
    )
    stub = WarmupStubProvider()
    router._providers["openai"] = stub  # noqa: SLF001

    entry = WarmupEntry(
        messages=[{"role": "user", "content": "hello world test warmup"}],
        model="gpt-4o-mini",
    )
    result1 = await router.warmup([entry])
    assert result1["warmed"] == 1
    assert result1["skipped"] == 0

    result2 = await router.warmup([entry], skip_cached=True)
    assert result2["skipped"] == 1
    assert result2["warmed"] == 0
