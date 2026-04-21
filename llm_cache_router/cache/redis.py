from __future__ import annotations

import asyncio
import json
import time
import uuid

import numpy as np
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_cache_router.cache.base import CacheBackend
from llm_cache_router.embeddings.encoder import HashingEncoder, SentenceEncoder
from llm_cache_router.models import CacheConfig, CacheEntry, LLMResponse

try:
    from redis.asyncio import from_url as redis_from_url
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover
    redis_from_url = None
    RedisError = Exception


class RedisSemanticCache(CacheBackend):
    def __init__(self, config: CacheConfig, redis_client=None) -> None:  # noqa: ANN001
        self._config = config
        self._namespace = config.redis_namespace
        self._index_key = f"{self._namespace}:entries"
        self._command_timeout_sec = config.redis_command_timeout_sec
        self._retry_attempts = config.redis_retry_attempts
        self._retry_backoff_sec = config.redis_retry_backoff_sec
        self._candidate_k = config.redis_candidate_k
        self._evictions = 0
        self._expired_removed = 0
        self._timeouts = 0
        self._retries = 0

        if config.embedding_model == "hash":
            self._encoder = HashingEncoder()
        else:
            try:
                self._encoder = SentenceEncoder(config.embedding_model)
            except Exception:  # pragma: no cover
                self._encoder = HashingEncoder()

        if redis_client is not None:
            self._redis = redis_client
            self._owns_client = False
            return

        if redis_from_url is None:  # pragma: no cover
            raise RuntimeError("redis dependency is required for RedisSemanticCache")
        self._redis = redis_from_url(config.redis_url, decode_responses=True)
        self._owns_client = True

    async def get(self, messages: list[dict[str, str]]) -> tuple[CacheEntry | None, float | None]:
        query_text = self._messages_to_text(messages)
        if len(query_text.strip()) < self._config.min_query_length:
            return None, None

        ids = await self._redis_call("zrange", self._index_key, 0, -1)
        if not ids:
            return None, None
        candidate_ids = self._pick_candidate_ids(ids)

        query_vec = self._encoder.encode(query_text)
        best_score = -1.0
        best_entry: CacheEntry | None = None
        stale_ids: list[str] = []
        raw_payloads = await self._redis_call(
            "mget", *[self._entry_key(entry_id) for entry_id in candidate_ids]
        )
        now_ts = time.time()
        for entry_id, raw in zip(candidate_ids, raw_payloads, strict=False):
            if not raw:
                stale_ids.append(entry_id)
                continue
            payload = json.loads(raw)
            entry = self._entry_from_payload(payload)
            if entry.is_expired(now_ts):
                stale_ids.append(entry_id)
                await self._redis_call("delete", self._entry_key(entry_id))
                self._expired_removed += 1
                continue
            score = self._cosine(query_vec, np.array(entry.embedding, dtype=np.float32))
            if score > best_score:
                best_score = score
                best_entry = entry

        if stale_ids:
            await self._redis_call("zrem", self._index_key, *stale_ids)

        if best_entry is None:
            return None, None
        if best_score < self._config.threshold:
            return None, best_score

        best_entry.hit_count += 1
        await self._persist_entry(best_entry, refresh_score=False)
        return best_entry, best_score

    async def set(self, messages: list[dict[str, str]], response: LLMResponse) -> None:
        query_text = self._messages_to_text(messages)
        if len(query_text.strip()) < self._config.min_query_length:
            return

        now_ts = time.time()
        entry = CacheEntry(
            query=query_text,
            response=response,
            embedding=self._encoder.encode(query_text).tolist(),
            created_at_ts=now_ts,
            ttl=self._config.ttl,
            hit_count=0,
        )
        await self._persist_entry(entry, refresh_score=True)
        await self._trim_max_entries()

    async def clear(self) -> None:
        ids = await self._redis_call("zrange", self._index_key, 0, -1)
        if ids:
            keys = [self._entry_key(entry_id) for entry_id in ids]
            await self._redis_call("delete", *keys)
        await self._redis_call("delete", self._index_key)

    async def close(self) -> None:
        if self._owns_client and hasattr(self._redis, "aclose"):
            await self._redis.aclose()

    async def _trim_max_entries(self) -> None:
        count = await self._redis_call("zcard", self._index_key)
        overflow = count - self._config.max_entries
        if overflow <= 0:
            return
        old_entries = await self._redis_call("zpopmin", self._index_key, overflow)
        if old_entries:
            keys = [self._entry_key(entry_id) for entry_id, _score in old_entries]
            await self._redis_call("delete", *keys)
            self._evictions += len(old_entries)

    async def _persist_entry(self, entry: CacheEntry, refresh_score: bool) -> None:
        entry_id = self._entry_id(entry)
        key = self._entry_key(entry_id)
        payload = {
            "query": entry.query,
            "embedding": entry.embedding,
            "created_at_ts": entry.created_at_ts,
            "ttl": entry.ttl,
            "hit_count": entry.hit_count,
            "response": entry.response.model_dump(),
        }
        await self._redis_call("set", key, json.dumps(payload), ex=entry.ttl)
        if refresh_score:
            await self._redis_call("zadd", self._index_key, {entry_id: entry.created_at_ts})
        else:
            score = await self._redis_call("zscore", self._index_key, entry_id)
            if score is None:
                await self._redis_call("zadd", self._index_key, {entry_id: entry.created_at_ts})

    def _entry_id(self, entry: CacheEntry) -> str:
        return f"{uuid.uuid5(uuid.NAMESPACE_DNS, entry.query)}"

    def _entry_key(self, entry_id: str) -> str:
        return f"{self._namespace}:entry:{entry_id}"

    def _pick_candidate_ids(self, ids: list[str]) -> list[str]:
        if self._candidate_k is None or self._candidate_k <= 0:
            return ids
        return ids[-self._candidate_k :]

    @staticmethod
    def _entry_from_payload(payload: dict) -> CacheEntry:
        return CacheEntry(
            query=payload["query"],
            response=LLMResponse.model_validate(payload["response"]),
            embedding=payload["embedding"],
            created_at_ts=float(payload["created_at_ts"]),
            ttl=int(payload["ttl"]),
            hit_count=int(payload.get("hit_count", 0)),
        )

    @staticmethod
    def _messages_to_text(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}:{content}")
        return "\n".join(parts).strip()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    async def _redis_call(self, method_name: str, *args, **kwargs):  # noqa: ANN002,ANN003,ANN202
        methods = {
            "zrange": self._redis.zrange,
            "mget": self._redis.mget,
            "delete": self._redis.delete,
            "zrem": self._redis.zrem,
            "zcard": self._redis.zcard,
            "zpopmin": self._redis.zpopmin,
            "set": self._redis.set,
            "zadd": self._redis.zadd,
            "zscore": self._redis.zscore,
        }
        if method_name not in methods:
            raise ValueError(f"Unsupported redis method: {method_name}")
        method = methods[method_name]
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max(1, self._retry_attempts)),
            wait=wait_exponential(multiplier=self._retry_backoff_sec, min=0.05, max=1.0),
            retry=retry_if_exception_type((RedisError, asyncio.TimeoutError, TimeoutError, OSError)),
            reraise=True,
        ):
            with attempt:
                try:
                    return await asyncio.wait_for(
                        method(*args, **kwargs),
                        timeout=self._command_timeout_sec,
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    self._timeouts += 1
                    raise
                except (RedisError, OSError):
                    self._retries += 1
                    raise

    def stats(self) -> dict[str, int]:
        return {
            "evictions": self._evictions,
            "expired_removed": self._expired_removed,
            "timeouts": self._timeouts,
            "retries": self._retries,
        }

