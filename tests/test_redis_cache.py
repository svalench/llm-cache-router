from __future__ import annotations

import asyncio
import time

import pytest

from llm_cache_router.cache.redis import RedisSemanticCache
from llm_cache_router.models import CacheConfig, LLMResponse


class FakeAsyncRedis:
    def __init__(self) -> None:
        self._kv: dict[str, tuple[str, float | None]] = {}
        self._zsets: dict[str, dict[str, float]] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        expires_at = (time.time() + ex) if ex is not None else None
        self._kv[key] = (value, expires_at)

    async def get(self, key: str) -> str | None:
        payload = self._kv.get(key)
        if payload is None:
            return None
        value, expires_at = payload
        if expires_at is not None and time.time() > expires_at:
            self._kv.pop(key, None)
            return None
        return value

    async def mget(self, *keys: str) -> list[str | None]:
        result: list[str | None] = []
        for key in keys:
            result.append(await self.get(key))
        return result

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self._kv.pop(key, None)
            self._zsets.pop(key, None)

    async def zadd(self, key: str, mapping: dict[str, float]) -> None:
        zset = self._zsets.setdefault(key, {})
        zset.update(mapping)

    async def zrange(self, key: str, start: int, end: int) -> list[str]:
        zset = self._zsets.get(key, {})
        items = sorted(zset.items(), key=lambda item: item[1])
        names = [name for name, _score in items]
        if end == -1:
            return names[start:]
        return names[start : end + 1]

    async def zrem(self, key: str, *members: str) -> None:
        zset = self._zsets.setdefault(key, {})
        for member in members:
            zset.pop(member, None)

    async def zcard(self, key: str) -> int:
        return len(self._zsets.get(key, {}))

    async def zpopmin(self, key: str, count: int) -> list[tuple[str, float]]:
        zset = self._zsets.setdefault(key, {})
        items = sorted(zset.items(), key=lambda item: item[1])[:count]
        for member, _score in items:
            zset.pop(member, None)
        return [(member, score) for member, score in items]

    async def zscore(self, key: str, member: str) -> float | None:
        return self._zsets.get(key, {}).get(member)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_redis_cache_hit_with_similarity() -> None:
    fake_redis = FakeAsyncRedis()
    cache = RedisSemanticCache(
        CacheConfig(
            backend="redis",
            threshold=0.7,
            min_query_length=1,
            embedding_model="hash",
            redis_namespace="test",
        ),
        redis_client=fake_redis,
    )
    messages = [{"role": "user", "content": "объясни маршрутизацию"}]
    response = LLMResponse(content="ok", provider_used="openai", model_used="gpt-4o-mini")

    await cache.set(messages, response)
    entry, similarity = await cache.get(messages)

    assert entry is not None
    assert similarity is not None
    assert similarity > 0.99


@pytest.mark.asyncio
async def test_redis_cache_ttl_expiration() -> None:
    fake_redis = FakeAsyncRedis()
    cache = RedisSemanticCache(
        CacheConfig(
            backend="redis",
            threshold=0.7,
            ttl=1,
            min_query_length=1,
            embedding_model="hash",
            redis_namespace="test_ttl",
        ),
        redis_client=fake_redis,
    )
    messages = [{"role": "user", "content": "короткий ttl"}]
    response = LLMResponse(content="ok", provider_used="openai", model_used="gpt-4o-mini")

    await cache.set(messages, response)
    await cache.get(messages)
    await asyncio.sleep(1.1)
    entry, _similarity = await cache.get(messages)

    assert entry is None


@pytest.mark.asyncio
async def test_redis_cache_respects_max_entries() -> None:
    fake_redis = FakeAsyncRedis()
    cache = RedisSemanticCache(
        CacheConfig(
            backend="redis",
            threshold=0.7,
            max_entries=2,
            min_query_length=1,
            embedding_model="hash",
            redis_namespace="test_size",
        ),
        redis_client=fake_redis,
    )

    await cache.set([{"role": "user", "content": "alpha one unique"}], LLMResponse(content="1", provider_used="ollama", model_used="llama3.2"))
    await cache.set([{"role": "user", "content": "bravo two unique"}], LLMResponse(content="2", provider_used="ollama", model_used="llama3.2"))
    await cache.set([{"role": "user", "content": "charlie three unique"}], LLMResponse(content="3", provider_used="ollama", model_used="llama3.2"))

    # Самая старая запись должна быть вытеснена.
    oldest, _ = await cache.get([{"role": "user", "content": "alpha one unique"}])
    newest, _ = await cache.get([{"role": "user", "content": "charlie three unique"}])

    assert oldest is None
    assert newest is not None
    stats = cache.stats()
    assert stats["evictions"] == 1


@pytest.mark.asyncio
async def test_redis_cache_candidate_k_limits_search_space() -> None:
    fake_redis = FakeAsyncRedis()
    cache = RedisSemanticCache(
        CacheConfig(
            backend="redis",
            threshold=0.7,
            max_entries=10,
            min_query_length=1,
            embedding_model="hash",
            redis_namespace="test_knn",
            redis_candidate_k=1,
        ),
        redis_client=fake_redis,
    )
    await cache.set([{"role": "user", "content": "older semantic request"}], LLMResponse(content="old", provider_used="ollama", model_used="llama3.2"))
    await cache.set([{"role": "user", "content": "newer but different"}], LLMResponse(content="new", provider_used="ollama", model_used="llama3.2"))
    entry, _ = await cache.get([{"role": "user", "content": "older semantic request"}])

    assert entry is None

