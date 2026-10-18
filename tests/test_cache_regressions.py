"""Regression tests for cache scopes and actual supported Qdrant client APIs."""

from __future__ import annotations

import pytest
from test_redis_cache import FakeAsyncRedis

from llm_cache_router.cache.memory import InMemorySemanticCache
from llm_cache_router.cache.qdrant import QdrantSemanticCache
from llm_cache_router.cache.redis import RedisSemanticCache
from llm_cache_router.models import CacheConfig, LLMResponse


def response(content: str) -> LLMResponse:
    return LLMResponse(content=content, provider_used="stub", model_used="test")


@pytest.mark.asyncio
async def test_redis_preserves_same_query_for_each_model_and_key_version():
    config = CacheConfig(embedding_model="hash", exact_match=True, min_query_length=1)
    cache = RedisSemanticCache(config, redis_client=FakeAsyncRedis())
    messages = [{"role": "user", "content": "shared prompt"}]
    for version in ("v1", "v2"):
        config.key_version = version
        for model in ("model-a", "model-b"):
            await cache.set(messages, response(f"{version}/{model}"), model=model)
    for version in ("v1", "v2"):
        config.key_version = version
        for model in ("model-a", "model-b"):
            entry, _ = await cache.get(messages, model=model)
            assert entry is not None
            assert entry.response.content == f"{version}/{model}"


@pytest.mark.asyncio
async def test_qdrant_real_client_custom_version_and_invalidation(monkeypatch):
    client = pytest.importorskip("qdrant_client").AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr("llm_cache_router.cache.qdrant.AsyncQdrantClient", lambda **kwargs: client)
    config = CacheConfig(
        embedding_model="hash", exact_match=True, min_query_length=1, key_version="custom-v2"
    )
    cache = QdrantSemanticCache(config)
    try:
        messages = [{"role": "user", "content": "actual qdrant query"}]
        await cache.set(messages, response("cached"), model="a")
        entry, _ = await cache.get(messages, model="a")
        assert entry is not None
        assert entry.key_version == "custom-v2"
        assert (await cache.get(messages, model="b"))[0] is None
        config.key_version = "custom-v3"
        assert (await cache.get(messages, model="a"))[0] is None
        assert await cache.invalidate(model="a") == 1
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_qdrant_invalidate_before_first_use(monkeypatch):
    client = pytest.importorskip("qdrant_client").AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr("llm_cache_router.cache.qdrant.AsyncQdrantClient", lambda **kwargs: client)
    cache = QdrantSemanticCache(CacheConfig(embedding_model="hash"))
    try:
        assert await cache.invalidate() == 0
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_exact_match_does_not_confuse_message_boundaries():
    cache = InMemorySemanticCache(
        CacheConfig(embedding_model="hash", exact_match=True, min_query_length=1)
    )
    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "private prior context"},
    ]
    injected = [{"role": "user", "content": "hello\nassistant:private prior context"}]
    await cache.set(conversation, response("private response"))
    assert (await cache.get(injected))[0] is None
    assert (await cache.get(conversation))[0] is not None


def test_hash_embeddings_are_stable_across_python_processes():
    import json
    import os
    import subprocess
    import sys

    script = (
        "import json; from llm_cache_router.embeddings.encoder import HashingEncoder; "
        "print(json.dumps(HashingEncoder(384).encode('persistent shared cache').tolist()))"
    )
    vectors = []
    for seed in ("1", "2"):
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            text=True,
        )
        vectors.append(json.loads(output))
    assert vectors[0] == vectors[1]
