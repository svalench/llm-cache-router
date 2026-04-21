from __future__ import annotations

from abc import ABC, abstractmethod

from llm_cache_router.models import CacheEntry, LLMResponse


class CacheBackend(ABC):
    @abstractmethod
    async def get(self, messages: list[dict[str, str]]) -> tuple[CacheEntry | None, float | None]:
        raise NotImplementedError

    @abstractmethod
    async def set(self, messages: list[dict[str, str]], response: LLMResponse) -> None:
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {}
