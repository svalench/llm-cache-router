from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from llm_cache_router.models import LLMResponse


@dataclass(slots=True)
class ProviderConfig:
    name: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0


class ProviderError(RuntimeError):
    pass


class LLMProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=self.config.timeout)

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _extract_text_from_messages(messages: list[dict[str, str]]) -> str:
        return "\n".join(m.get("content", "") for m in messages if m.get("content"))

