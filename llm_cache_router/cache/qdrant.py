from __future__ import annotations

from llm_cache_router.cache.base import CacheBackend
from llm_cache_router.models import CacheEntry, LLMResponse


class QdrantSemanticCache(CacheBackend):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        raise NotImplementedError("Qdrant backend is planned for v0.5")

    async def get(self, messages: list[dict[str, str]]) -> tuple[CacheEntry | None, float | None]:
        raise NotImplementedError

    async def set(self, messages: list[dict[str, str]], response: LLMResponse) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

