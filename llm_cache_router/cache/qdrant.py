from __future__ import annotations

import asyncio
import importlib
import logging
import time
from typing import Any
from uuid import uuid4

from llm_cache_router.cache.base import CacheBackend
from llm_cache_router.embeddings.encoder import EncoderProtocol, HashingEncoder, SentenceEncoder
from llm_cache_router.models import CacheConfig, CacheEntry, LLMResponse, Message

_qdrant_client: Any | None
_qdrant_models: Any | None
try:
    _qdrant_client = importlib.import_module("qdrant_client")
    _qdrant_models = importlib.import_module("qdrant_client.http.models")
except ImportError:  # pragma: no cover
    _qdrant_client = None
    _qdrant_models = None

if _qdrant_client is None or _qdrant_models is None:  # pragma: no cover
    AsyncQdrantClient: Any = None
    Distance: Any = None
    Direction: Any = None
    FieldCondition: Any = None
    Filter: Any = None
    FilterSelector: Any = None
    MatchValue: Any = None
    OrderBy: Any = None
    PointIdsList: Any = None
    PointStruct: Any = None
    VectorParams: Any = None
else:
    AsyncQdrantClient = _qdrant_client.AsyncQdrantClient
    Distance = _qdrant_models.Distance
    Direction = _qdrant_models.Direction
    FieldCondition = _qdrant_models.FieldCondition
    Filter = _qdrant_models.Filter
    FilterSelector = _qdrant_models.FilterSelector
    MatchValue = _qdrant_models.MatchValue
    OrderBy = _qdrant_models.OrderBy
    PointIdsList = _qdrant_models.PointIdsList
    PointStruct = _qdrant_models.PointStruct
    VectorParams = _qdrant_models.VectorParams

logger = logging.getLogger(__name__)


class QdrantSemanticCache(CacheBackend):
    def __init__(self, config: CacheConfig) -> None:
        if (
            AsyncQdrantClient is None or VectorParams is None or Distance is None
        ):  # pragma: no cover
            raise RuntimeError(
                "qdrant-client is required for QdrantSemanticCache. "
                "Install with: pip install 'llm-cache-router[qdrant]'"
            )
        self._config = config
        self._dimension = 384
        self._collection_name = config.qdrant_collection
        self._client = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        self._init_lock = asyncio.Lock()
        self._collection_ready = False
        self._total_vectors = 0

        self._encoder: EncoderProtocol
        if config.embedding_model == "hash":
            self._encoder = HashingEncoder(self._dimension)
        else:
            try:
                self._encoder = SentenceEncoder(config.embedding_model)
            except Exception:  # pragma: no cover
                self._encoder = HashingEncoder(self._dimension)

    async def get(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
    ) -> tuple[CacheEntry | None, float | None]:
        query_text = self._messages_to_text(messages)
        if len(query_text.strip()) < self._config.min_query_length:
            return None, None

        await self._ensure_collection()
        embedding = self._encoder.encode(query_text).tolist()
        qdrant_client: Any = self._client
        search_kwargs: dict[str, Any] = {
            "collection_name": self._collection_name,
            "query_vector": embedding,
            "limit": 1,
            "with_payload": True,
            "score_threshold": self._config.threshold,
        }
        query_filter = self._build_query_filter(model, query_text)
        if query_filter is not None:
            search_kwargs["query_filter"] = query_filter
        results = await qdrant_client.search(**search_kwargs)
        if not results:
            return None, None

        top = results[0]
        score = float(top.score)
        if score < self._config.threshold:
            return None, score

        payload = top.payload or {}
        entry = self._entry_from_payload(payload)
        if entry is None:
            return None, score
        if entry.is_expired(time.time()):
            return None, score

        entry.hit_count += 1
        await self._client.set_payload(
            collection_name=self._collection_name,
            payload={"hit_count": entry.hit_count},
            points=[top.id],
        )
        return entry, score

    async def set(
        self,
        messages: list[Message],
        response: LLMResponse,
        *,
        model: str | None = None,
    ) -> None:
        query_text = self._messages_to_text(messages)
        if len(query_text.strip()) < self._config.min_query_length:
            return

        await self._ensure_collection()
        embedding = self._encoder.encode(query_text)
        entry = CacheEntry(
            query=query_text,
            response=response,
            embedding=embedding.tolist(),
            created_at_ts=time.time(),
            ttl=self._config.ttl,
            hit_count=0,
            model=model,
        )
        payload = {
            "query": entry.query,
            "response": entry.response.model_dump(),
            "created_at_ts": entry.created_at_ts,
            "ttl": entry.ttl,
            "hit_count": entry.hit_count,
            "model": entry.model,
            "key_version": entry.key_version,
        }

        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            ],
        )
        self._total_vectors = await self._count_vectors()
        if self._config.max_entries > 0 and self._total_vectors > self._config.max_entries:
            overflow = self._total_vectors - self._config.max_entries
            await self._evict_oldest(overflow)
            self._total_vectors = await self._count_vectors()

    async def clear(self) -> None:
        await self._ensure_collection()
        await self._client.delete_collection(collection_name=self._collection_name)
        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._dimension,
                distance=Distance.COSINE,
            ),
        )
        self._total_vectors = 0

    async def invalidate(self, *, model: str | None = None) -> int:
        if model is None:
            count = await self._count_vectors()
            await self.clear()
            return count
        await self._ensure_collection()
        if Filter is None or FieldCondition is None or MatchValue is None or FilterSelector is None:
            raise NotImplementedError("qdrant-client is required for selective invalidation")
        total_before = await self._count_vectors()
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="model",
                            match=MatchValue(value=model),
                        )
                    ]
                )
            ),
        )
        self._total_vectors = await self._count_vectors()
        return total_before - self._total_vectors

    async def close(self) -> None:
        await self._client.close()

    def stats(self) -> dict[str, int]:
        return {"total_vectors": self._total_vectors}

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        async with self._init_lock:
            if self._collection_ready:
                return
            exists = await self._client.collection_exists(collection_name=self._collection_name)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._dimension,
                        distance=Distance.COSINE,
                    ),
                )
            self._total_vectors = await self._count_vectors()
            self._collection_ready = True

    async def _count_vectors(self) -> int:
        count = await self._client.count(collection_name=self._collection_name, exact=True)
        return int(count.count)

    async def _evict_oldest(self, count: int) -> None:
        if count <= 0:
            return
        results, _ = await self._client.scroll(
            collection_name=self._collection_name,
            limit=count,
            with_payload=["created_at_ts"],
            with_vectors=False,
            order_by=OrderBy(
                key="created_at_ts",
                direction=Direction.ASC,
            ),
        )
        ids_to_delete = [point.id for point in results]
        if ids_to_delete:
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=ids_to_delete),
            )
            logger.info("Qdrant evicted %d oldest entries", len(ids_to_delete))

    @staticmethod
    def _entry_from_payload(payload: dict) -> CacheEntry | None:
        try:
            return CacheEntry(
                query=payload["query"],
                response=LLMResponse.model_validate(payload["response"]),
                embedding=[],
                created_at_ts=float(payload["created_at_ts"]),
                ttl=int(payload["ttl"]),
                hit_count=int(payload.get("hit_count", 0)),
                model=payload.get("model"),
                key_version=str(payload.get("key_version", "v1")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _build_query_filter(self, model: str | None, query_text: str | None) -> Any | None:
        if Filter is None or FieldCondition is None or MatchValue is None:
            return None
        must: list[Any] = [
            FieldCondition(
                key="key_version",
                match=MatchValue(value=self._config.key_version),
            )
        ]
        if model is not None:
            must.append(
                FieldCondition(
                    key="model",
                    match=MatchValue(value=model),
                )
            )
        if self._config.exact_match and query_text is not None:
            must.append(
                FieldCondition(
                    key="query",
                    match=MatchValue(value=query_text),
                )
            )
        return Filter(must=must)
