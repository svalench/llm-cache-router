from __future__ import annotations

import asyncio
import heapq
import time

import numpy as np

from llm_cache_router.cache.base import CacheBackend
from llm_cache_router.embeddings.encoder import EncoderProtocol, HashingEncoder, SentenceEncoder
from llm_cache_router.models import CacheConfig, CacheEntry, LLMResponse, Message

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class InMemorySemanticCache(CacheBackend):
    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()
        self._entries: list[CacheEntry] = []
        self._faiss_index = None
        self._vectors: list[np.ndarray] = []
        self._dimension = 384
        self._use_faiss = faiss is not None

        self._encoder: EncoderProtocol
        if config.embedding_model == "hash":
            self._encoder = HashingEncoder(self._dimension)
        else:
            try:
                self._encoder = SentenceEncoder(config.embedding_model)
            except Exception:  # pragma: no cover
                self._encoder = HashingEncoder(self._dimension)

        if self._use_faiss:
            self._faiss_index = faiss.IndexFlatIP(self._dimension)
        self._evictions = 0
        self._expired_removed = 0

    async def get(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
    ) -> tuple[CacheEntry | None, float | None]:
        query_text = self._messages_to_text(messages)
        if len(query_text.strip()) < self._config.min_query_length:
            return None, None
        if not self._entries:
            return None, None

        embedding = self._encoder.encode(query_text)
        async with self._lock:
            now_ts = time.time()
            self._purge_expired(now_ts)
            if not self._entries:
                return None, None

            score, idx = self._search_top1(
                embedding,
                model=model,
                query_text=query_text,
            )
            if idx < 0:
                return None, None
            if score < self._config.threshold:
                return None, score

            entry = self._entries[idx]
            entry.hit_count += 1
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
        embedding = self._encoder.encode(query_text)
        entry = CacheEntry(
            query=query_text,
            response=response,
            embedding=embedding.tolist(),
            created_at_ts=time.time(),
            ttl=self._config.ttl,
            hit_count=0,
            model=model,
            key_version=self._config.key_version,
        )

        async with self._lock:
            self._entries.append(entry)
            self._vectors.append(embedding.astype(np.float32))
            if self._use_faiss and self._faiss_index is not None:
                self._faiss_index.add(np.array([embedding], dtype=np.float32))

            if len(self._entries) > self._config.max_entries:
                self._evict_lfu()

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._vectors.clear()
            self._rebuild_index()

    async def invalidate(self, *, model: str | None = None) -> int:
        async with self._lock:
            kept: list[tuple[CacheEntry, np.ndarray]] = []
            removed = 0
            for entry, vec in zip(self._entries, self._vectors, strict=False):
                if model is None or entry.model == model:
                    removed += 1
                else:
                    kept.append((entry, vec))
            if removed:
                self._entries = [pair[0] for pair in kept]
                self._vectors = [pair[1] for pair in kept]
                self._rebuild_index()
            return removed

    def _entry_visible(
        self,
        entry: CacheEntry,
        *,
        model: str | None,
        query_text: str | None = None,
    ) -> bool:
        if entry.key_version != self._config.key_version:
            return False
        if not self._model_matches(entry.model, model):
            return False
        if self._config.exact_match and query_text is not None and entry.query != query_text:
            return False
        return True

    def _search_top1(
        self,
        embedding: np.ndarray,
        *,
        model: str | None,
        query_text: str | None = None,
    ) -> tuple[float, int]:
        best_score = -1.0
        best_idx = -1

        if self._use_faiss and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            k = min(self._faiss_index.ntotal, len(self._entries))
            scores, indices = self._faiss_index.search(
                np.array([embedding], dtype=np.float32),
                k=k,
            )
            for raw_score, raw_idx in zip(scores[0], indices[0], strict=False):
                idx = int(raw_idx)
                if idx < 0 or idx >= len(self._entries):
                    continue
                entry = self._entries[idx]
                if not self._entry_visible(entry, model=model, query_text=query_text):
                    continue
                score = float(raw_score)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_score, best_idx

        matrix = np.vstack(self._vectors).astype(np.float32)
        scores = matrix @ embedding
        for idx, score in enumerate(scores):
            entry = self._entries[idx]
            if not self._entry_visible(entry, model=model, query_text=query_text):
                continue
            value = float(score)
            if value > best_score:
                best_score = value
                best_idx = idx
        return best_score, best_idx

    def _purge_expired(self, now_ts: float) -> None:
        old_size = len(self._entries)
        alive_pairs = [
            (entry, vec)
            for entry, vec in zip(self._entries, self._vectors, strict=False)
            if not entry.is_expired(now_ts)
        ]
        if len(alive_pairs) == len(self._entries):
            return
        self._expired_removed += old_size - len(alive_pairs)
        self._entries = [item[0] for item in alive_pairs]
        self._vectors = [item[1] for item in alive_pairs]
        self._rebuild_index()

    def stats(self) -> dict[str, int]:
        return {
            "evictions": self._evictions,
            "expired_removed": self._expired_removed,
        }

    def _rebuild_index(self) -> None:
        if not self._use_faiss:
            return
        self._faiss_index = faiss.IndexFlatIP(self._dimension)
        if self._vectors:
            self._faiss_index.add(np.array(self._vectors, dtype=np.float32))

    def _evict_lfu(self) -> None:
        overflow = len(self._entries) - self._config.max_entries
        if overflow <= 0:
            return
        heap = [
            (entry.hit_count, entry.created_at_ts, idx) for idx, entry in enumerate(self._entries)
        ]
        heapq.heapify(heap)
        to_remove: set[int] = set()
        for _ in range(overflow):
            _, _, idx = heapq.heappop(heap)
            to_remove.add(idx)
        self._evictions += len(to_remove)
        survived = [
            (entry, vec)
            for i, (entry, vec) in enumerate(zip(self._entries, self._vectors, strict=False))
            if i not in to_remove
        ]
        self._entries = [pair[0] for pair in survived]
        self._vectors = [pair[1] for pair in survived]
        self._rebuild_index()
