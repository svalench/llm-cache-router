from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from test_redis_cache import FakeAsyncRedis

import llm_cache_router.cache.qdrant as qdrant_module
from llm_cache_router.cache.memory import InMemorySemanticCache
from llm_cache_router.cache.redis import RedisSemanticCache
from llm_cache_router.models import CacheConfig, LLMResponse, Message


def _response(content: str = "ok", model: str = "gpt-4o-mini") -> LLMResponse:
    return LLMResponse(content=content, provider_used="openai", model_used=model)


def _messages(text: str) -> list[Message]:
    return [{"role": "user", "content": text}]


# --- In-memory: invalidate ---


@pytest.mark.asyncio
async def test_memory_invalidate_by_model() -> None:
    cache = InMemorySemanticCache(
        CacheConfig(backend="memory", threshold=0.99, min_query_length=1, embedding_model="hash")
    )
    await cache.set(_messages("вопрос про кэш один"), _response(), model="gpt-4o-mini")
    await cache.set(_messages("вопрос про кэш два"), _response(), model="gpt-4o")
    await cache.set(_messages("вопрос про кэш три"), _response(), model="gpt-4o-mini")

    removed = await cache.invalidate(model="gpt-4o-mini")

    assert removed == 2
    remaining, _ = await cache.get(_messages("вопрос про кэш два"), model="gpt-4o")
    assert remaining is not None
    gone, _ = await cache.get(_messages("вопрос про кэш один"), model="gpt-4o-mini")
    assert gone is None


@pytest.mark.asyncio
async def test_memory_invalidate_all() -> None:
    cache = InMemorySemanticCache(
        CacheConfig(backend="memory", threshold=0.99, min_query_length=1, embedding_model="hash")
    )
    await cache.set(_messages("вопрос про кэш один"), _response(), model="gpt-4o-mini")
    await cache.set(_messages("вопрос про кэш два"), _response(), model="gpt-4o")

    removed = await cache.invalidate()

    assert removed == 2
    entry, _ = await cache.get(_messages("вопрос про кэш один"), model="gpt-4o-mini")
    assert entry is None


@pytest.mark.asyncio
async def test_memory_invalidate_empty_cache() -> None:
    cache = InMemorySemanticCache(
        CacheConfig(backend="memory", min_query_length=1, embedding_model="hash")
    )

    assert await cache.invalidate(model="gpt-4o") == 0
    assert await cache.invalidate() == 0


# --- In-memory: exact_match ---


@pytest.mark.asyncio
async def test_memory_exact_match_returns_only_identical_query() -> None:
    cache = InMemorySemanticCache(
        CacheConfig(
            backend="memory",
            threshold=0.5,  # низкий порог: без exact_match это был бы хит
            min_query_length=1,
            embedding_model="hash",
            exact_match=True,
        )
    )
    original = "как сбросить пароль пользователя"
    await cache.set(_messages(original), _response())

    exact, similarity = await cache.get(_messages(original))
    assert exact is not None
    assert similarity is not None

    similar = "как сбросить пароль пользователя администратора"
    missed, _ = await cache.get(_messages(similar))
    assert missed is None


@pytest.mark.asyncio
async def test_memory_semantic_mode_matches_similar_query() -> None:
    cache = InMemorySemanticCache(
        CacheConfig(
            backend="memory",
            threshold=0.5,
            min_query_length=1,
            embedding_model="hash",
        )
    )
    await cache.set(_messages("как сбросить пароль пользователя"), _response())

    # Keep the first word: extracted cache text prefixes it with "user:".
    # Reorder the remaining words to preserve the actual hashed token set.
    similar, _ = await cache.get(_messages("как пароль пользователя сбросить"))

    assert similar is not None


# --- In-memory: key_version ---


@pytest.mark.asyncio
async def test_memory_key_version_busts_old_entries() -> None:
    config = CacheConfig(
        backend="memory",
        threshold=0.99,
        min_query_length=1,
        embedding_model="hash",
        key_version="v1",
    )
    cache = InMemorySemanticCache(config)
    await cache.set(_messages("вопрос про инвалидацию кэша"), _response())

    # Новый деплой промпта -> новая версия ключей, старые записи невидимы
    config.key_version = "v2-prompt-update"
    entry, _ = await cache.get(_messages("вопрос про инвалидацию кэша"))
    assert entry is None

    # Но записи по-прежнему физически в кэше и удаляются полной инвалидацией
    assert await cache.invalidate() == 1


# --- Redis: invalidate / exact_match / key_version ---


def _redis_cache(config: CacheConfig) -> RedisSemanticCache:
    return RedisSemanticCache(config, redis_client=FakeAsyncRedis())


@pytest.mark.asyncio
async def test_redis_invalidate_by_model() -> None:
    cache = _redis_cache(
        CacheConfig(
            backend="redis",
            threshold=0.99,
            min_query_length=1,
            embedding_model="hash",
        )
    )
    await cache.set(_messages("вопрос про redis кэш один"), _response(), model="gpt-4o-mini")
    await cache.set(_messages("вопрос про redis кэш два"), _response(), model="gpt-4o")

    removed = await cache.invalidate(model="gpt-4o-mini")

    assert removed == 1
    kept, _ = await cache.get(_messages("вопрос про redis кэш два"), model="gpt-4o")
    assert kept is not None
    gone, _ = await cache.get(_messages("вопрос про redis кэш один"), model="gpt-4o-mini")
    assert gone is None


@pytest.mark.asyncio
async def test_redis_exact_match_and_key_version() -> None:
    config = CacheConfig(
        backend="redis",
        threshold=0.5,
        min_query_length=1,
        embedding_model="hash",
        exact_match=True,
        key_version="v1",
    )
    cache = _redis_cache(config)
    original = "как удалить аккаунт навсегда"
    await cache.set(_messages(original), _response())

    exact, _ = await cache.get(_messages(original))
    assert exact is not None

    missed, _ = await cache.get(_messages("как удалить аккаунт навсегда и все данные"))
    assert missed is None

    config.key_version = "v2"
    stale, _ = await cache.get(_messages(original))
    assert stale is None
    assert await cache.invalidate() == 1


# --- Qdrant: invalidate / key_version ---


class FakeQdrantFilterClient:
    def __init__(self, url: str, api_key: str | None = None) -> None:
        del url, api_key
        self.points: list[dict] = []

    async def collection_exists(self, collection_name: str) -> bool:
        del collection_name
        return True

    async def create_collection(self, collection_name: str, vectors_config: object) -> None:
        del collection_name, vectors_config

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool,
        score_threshold: float | None = None,
        query_filter: dict | None = None,
    ) -> list[SimpleNamespace]:
        del collection_name, with_payload
        ranked: list[tuple[float, dict]] = []
        for point in self.points:
            if query_filter is not None and not self._matches(query_filter, point["payload"]):
                continue
            score = self._cosine(query_vector, point["vector"])
            if score_threshold is not None and score < score_threshold:
                continue
            ranked.append((score, point))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            SimpleNamespace(id=point["id"], score=score, payload=point["payload"])
            for score, point in ranked[:limit]
        ]

    async def set_payload(self, collection_name: str, payload: dict, points: list[str]) -> None:
        del collection_name
        point_ids = set(points)
        for point in self.points:
            if point["id"] in point_ids:
                point["payload"].update(payload)

    async def upsert(self, collection_name: str, points: list) -> None:
        del collection_name
        for point in points:
            self.points.append(
                {"id": point.id, "vector": point.vector, "payload": dict(point.payload)}
            )

    async def delete(self, collection_name: str, points_selector: object) -> None:
        del collection_name
        points = getattr(points_selector, "points", None)
        if points is not None:
            point_ids = set(points)
            self.points = [p for p in self.points if p["id"] not in point_ids]
            return
        # dict-based FilterSelector: {"filter": {"must": [{"key": ..., "match": {"value": ...}}]}}
        conditions = points_selector["filter"]["must"]  # type: ignore[index]
        self.points = [
            p for p in self.points if not self._conditions_match(conditions, p["payload"])
        ]

    async def delete_collection(self, collection_name: str) -> None:
        del collection_name
        self.points = []

    async def count(self, collection_name: str, exact: bool = True) -> SimpleNamespace:
        del collection_name, exact
        return SimpleNamespace(count=len(self.points))

    async def close(self) -> None:
        return None

    @staticmethod
    def _matches(query_filter: dict, payload: dict) -> bool:
        return FakeQdrantFilterClient._conditions_match(query_filter.get("must", []), payload)

    @staticmethod
    def _conditions_match(conditions: list, payload: dict) -> bool:
        return all(payload.get(cond["key"]) == cond["match"]["value"] for cond in conditions)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def _patch_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qdrant_module, "AsyncQdrantClient", FakeQdrantFilterClient)
    monkeypatch.setattr(qdrant_module, "Distance", type("D", (), {"COSINE": "cosine"}))
    monkeypatch.setattr(
        qdrant_module,
        "VectorParams",
        lambda size, distance: SimpleNamespace(size=size, distance=distance),
    )
    monkeypatch.setattr(
        qdrant_module,
        "PointStruct",
        lambda id, vector, payload: SimpleNamespace(id=id, vector=vector, payload=payload),
    )  # noqa: A002
    monkeypatch.setattr(qdrant_module, "Filter", lambda must: {"must": must})
    monkeypatch.setattr(
        qdrant_module, "FieldCondition", lambda key, match: {"key": key, "match": match}
    )
    monkeypatch.setattr(qdrant_module, "MatchValue", lambda value: {"value": value})
    monkeypatch.setattr(qdrant_module, "FilterSelector", lambda filter: {"filter": filter})  # noqa: A002


@pytest.mark.asyncio
async def test_qdrant_invalidate_by_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_qdrant(monkeypatch)
    cache = qdrant_module.QdrantSemanticCache(
        CacheConfig(
            backend="qdrant",
            threshold=0.99,
            min_query_length=1,
            embedding_model="hash",
            qdrant_collection="invalidate_test",
        )
    )
    await cache.set(_messages("вопрос про qdrant кэш один"), _response(), model="gpt-4o-mini")
    await cache.set(_messages("вопрос про qdrant кэш два"), _response(), model="gpt-4o")

    removed = await cache.invalidate(model="gpt-4o-mini")

    assert removed == 1
    kept, _ = await cache.get(_messages("вопрос про qdrant кэш два"), model="gpt-4o")
    assert kept is not None
    gone, _ = await cache.get(_messages("вопрос про qdrant кэш один"), model="gpt-4o-mini")
    assert gone is None


@pytest.mark.asyncio
async def test_qdrant_invalidate_all(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_qdrant(monkeypatch)
    cache = qdrant_module.QdrantSemanticCache(
        CacheConfig(
            backend="qdrant",
            min_query_length=1,
            embedding_model="hash",
            qdrant_collection="invalidate_all_test",
        )
    )
    await cache.set(_messages("вопрос про qdrant кэш один"), _response(), model="gpt-4o-mini")
    await cache.set(_messages("вопрос про qdrant кэш два"), _response(), model="gpt-4o")

    assert await cache.invalidate() == 2
    entry, _ = await cache.get(_messages("вопрос про qdrant кэш один"), model="gpt-4o-mini")
    assert entry is None


@pytest.mark.asyncio
async def test_qdrant_key_version_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_qdrant(monkeypatch)
    config = CacheConfig(
        backend="qdrant",
        threshold=0.99,
        min_query_length=1,
        embedding_model="hash",
        qdrant_collection="key_version_test",
        key_version="v1",
    )
    cache = qdrant_module.QdrantSemanticCache(config)
    await cache.set(_messages("вопрос про версии ключей"), _response())

    hit, _ = await cache.get(_messages("вопрос про версии ключей"))
    assert hit is not None

    config.key_version = "v2-deploy"
    stale, _ = await cache.get(_messages("вопрос про версии ключей"))
    assert stale is None


@pytest.mark.asyncio
async def test_qdrant_exact_match(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_qdrant(monkeypatch)
    cache = qdrant_module.QdrantSemanticCache(
        CacheConfig(
            backend="qdrant",
            threshold=0.5,
            min_query_length=1,
            embedding_model="hash",
            exact_match=True,
            qdrant_collection="exact_match_test",
        )
    )
    original = "как обновить billing данные"
    await cache.set(_messages(original), _response())

    exact, _ = await cache.get(_messages(original))
    assert exact is not None

    missed, _ = await cache.get(_messages("как обновить billing данные компании"))
    assert missed is None


@pytest.mark.asyncio
async def test_router_invalidate_cache_delegates_to_backend() -> None:
    from llm_cache_router.router import LLMRouter

    router = LLMRouter(
        providers={"openai": {"api_key": "test", "models": ["gpt-4o-mini"]}},
        cache=CacheConfig(backend="memory", min_query_length=1, embedding_model="hash"),
    )
    await router._cache.set(_messages("вопрос про роутер кэш"), _response(), model="gpt-4o-mini")

    removed = await router.invalidate_cache(model="gpt-4o-mini")

    assert removed == 1
    await router.clear_cache()
    assert await router.invalidate_cache() == 0
    await router.close()
