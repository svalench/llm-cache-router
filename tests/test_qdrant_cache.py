from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import llm_cache_router.cache.qdrant as qdrant_module
from llm_cache_router.models import CacheConfig, LLMResponse


class FakeDistance:
    COSINE = "cosine"


class FakeVectorParams:
    def __init__(self, size: int, distance: str) -> None:
        self.size = size
        self.distance = distance


class FakePointStruct:
    def __init__(self, id: str, vector: list[float], payload: dict) -> None:  # noqa: A002
        self.id = id
        self.vector = vector
        self.payload = payload


class FakeAsyncQdrantClient:
    def __init__(self, url: str, api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key
        self.collections: dict[str, list[dict]] = {}
        self.closed = False

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, vectors_config: FakeVectorParams) -> None:
        del vectors_config
        self.collections.setdefault(collection_name, [])

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool,
        score_threshold: float | None = None,
    ) -> list[SimpleNamespace]:
        del with_payload
        points = self.collections.get(collection_name, [])
        ranked: list[tuple[float, dict]] = []
        for point in points:
            score = self._cosine(query_vector, point["vector"])
            if score_threshold is not None and score < score_threshold:
                continue
            ranked.append((score, point))
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_items = ranked[:limit]
        return [
            SimpleNamespace(id=point["id"], score=score, payload=point["payload"])
            for score, point in top_items
        ]

    async def set_payload(self, collection_name: str, payload: dict, points: list[str]) -> None:
        point_ids = set(points)
        for point in self.collections.get(collection_name, []):
            if point["id"] in point_ids:
                point["payload"].update(payload)

    async def upsert(self, collection_name: str, points: list[FakePointStruct]) -> None:
        storage = self.collections.setdefault(collection_name, [])
        for point in points:
            storage.append({"id": point.id, "vector": point.vector, "payload": dict(point.payload)})

    async def delete_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)

    async def count(self, collection_name: str, exact: bool = True) -> SimpleNamespace:
        del exact
        return SimpleNamespace(count=len(self.collections.get(collection_name, [])))

    async def close(self) -> None:
        self.closed = True

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


@pytest.mark.asyncio
async def test_qdrant_cache_unit_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qdrant_module, "AsyncQdrantClient", FakeAsyncQdrantClient)
    monkeypatch.setattr(qdrant_module, "Distance", FakeDistance)
    monkeypatch.setattr(qdrant_module, "VectorParams", FakeVectorParams)
    monkeypatch.setattr(qdrant_module, "PointStruct", FakePointStruct)

    cache = qdrant_module.QdrantSemanticCache(
        CacheConfig(
            backend="qdrant",
            threshold=0.7,
            min_query_length=1,
            embedding_model="hash",
            qdrant_collection="test_qdrant_cache",
        )
    )
    messages = [{"role": "user", "content": "qdrant cache test"}]
    response = LLMResponse(content="ok", provider_used="openai", model_used="gpt-4o-mini")

    await cache.set(messages, response)
    entry, score = await cache.get(messages)

    assert entry is not None
    assert score is not None
    assert score >= 0.99
    assert entry.hit_count == 1
    assert cache.stats()["total_vectors"] == 1

    await cache.clear()
    cleared_entry, _ = await cache.get(messages)
    assert cleared_entry is None
    assert cache.stats()["total_vectors"] == 0

    await cache.close()
    assert cache._client.closed is True
