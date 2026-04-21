from __future__ import annotations

import time

import numpy as np

from llm_cache_router.cache.base import CacheBackend
from llm_cache_router.embeddings.encoder import HashingEncoder, SentenceEncoder
from llm_cache_router.models import CacheConfig, CacheEntry, LLMResponse

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class InMemorySemanticCache(CacheBackend):
    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._entries: list[CacheEntry] = []
        self._faiss_index = None
        self._vectors: list[np.ndarray] = []
        self._dimension = 384
        self._use_faiss = faiss is not None

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

    async def get(self, messages: list[dict[str, str]]) -> tuple[CacheEntry | None, float | None]:
        query_text = self._messages_to_text(messages)
        if len(query_text.strip()) < self._config.min_query_length:
            return None, None
        if not self._entries:
            return None, None

        embedding = self._encoder.encode(query_text)
        now_ts = time.time()
        self._purge_expired(now_ts)
        if not self._entries:
            return None, None

        score, idx = self._search_top1(embedding)
        if idx < 0:
            return None, None
        if score < self._config.threshold:
            return None, score

        entry = self._entries[idx]
        entry.hit_count += 1
        return entry, score

    async def set(self, messages: list[dict[str, str]], response: LLMResponse) -> None:
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
        )

        self._entries.append(entry)
        self._vectors.append(embedding.astype(np.float32))
        if self._use_faiss and self._faiss_index is not None:
            self._faiss_index.add(np.array([embedding], dtype=np.float32))

        if len(self._entries) > self._config.max_entries:
            overflow = len(self._entries) - self._config.max_entries
            self._evictions += overflow
            self._entries = self._entries[-self._config.max_entries :]
            self._vectors = self._vectors[-self._config.max_entries :]
            self._rebuild_index()

    async def clear(self) -> None:
        self._entries.clear()
        self._vectors.clear()
        self._rebuild_index()

    def _search_top1(self, embedding: np.ndarray) -> tuple[float, int]:
        if self._use_faiss and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            scores, indices = self._faiss_index.search(np.array([embedding], dtype=np.float32), k=1)
            return float(scores[0][0]), int(indices[0][0])

        matrix = np.vstack(self._vectors).astype(np.float32)
        scores = matrix @ embedding
        idx = int(np.argmax(scores))
        return float(scores[idx]), idx

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

    @staticmethod
    def _messages_to_text(messages: list[dict[str, str]]) -> str:
        chunks = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chunks.append(f"{role}:{content}")
        return "\n".join(chunks).strip()

