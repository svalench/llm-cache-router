from __future__ import annotations

import asyncio

import pytest

from llm_cache_router.cache.memory import InMemorySemanticCache
from llm_cache_router.models import CacheConfig, LLMResponse


@pytest.mark.asyncio
async def test_memory_cache_hit_and_similarity() -> None:
    cache = InMemorySemanticCache(
        CacheConfig(backend="memory", threshold=0.7, min_query_length=1, embedding_model="hash")
    )
    messages = [{"role": "user", "content": "объясни семантический кэш"}]
    response = LLMResponse(
        content="семантический кэш - это ...",
        provider_used="openai",
        model_used="gpt-4o-mini",
    )

    await cache.set(messages, response)
    entry, similarity = await cache.get(messages)

    assert entry is not None
    assert similarity is not None
    assert similarity >= 0.99


@pytest.mark.asyncio
async def test_memory_cache_skips_too_short_queries() -> None:
    cache = InMemorySemanticCache(
        CacheConfig(backend="memory", threshold=0.7, min_query_length=10, embedding_model="hash")
    )
    messages = [{"role": "user", "content": "hi"}]
    response = LLMResponse(content="hello", provider_used="ollama", model_used="llama3.2")

    await cache.set(messages, response)
    entry, similarity = await cache.get(messages)

    assert entry is None
    assert similarity is None


@pytest.mark.asyncio
async def test_concurrent_set_get() -> None:
    cache = InMemorySemanticCache(CacheConfig(embedding_model="hash"))
    response = LLMResponse(content="test", provider_used="p", model_used="m")
    messages = [{"role": "user", "content": "hello world test concurrent"}]

    set_tasks = [asyncio.create_task(cache.set(messages, response)) for _ in range(20)]
    await asyncio.gather(*set_tasks)
    get_tasks = [asyncio.create_task(cache.get(messages)) for _ in range(20)]
    await asyncio.gather(*get_tasks)

    assert len(cache._entries) <= cache._config.max_entries

