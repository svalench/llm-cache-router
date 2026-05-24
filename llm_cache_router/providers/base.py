from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import httpx

from llm_cache_router.models import LLMResponse, LLMStreamChunk, Message
from llm_cache_router.retry import RetryConfig


@dataclass(slots=True)
class ProviderConfig:
    name: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    retry: RetryConfig = field(default_factory=RetryConfig)


class ProviderError(RuntimeError):
    pass


class LLMProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=self.config.timeout)

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    async def close(self) -> None:
        await self._client.aclose()

    async def stream(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        response = await self.complete(messages, model, temperature, max_tokens)
        yield LLMStreamChunk(
            delta=response.content,
            provider_used=response.provider_used,
            model_used=response.model_used,
            is_final=True,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )

    @classmethod
    def _extract_text_from_messages(cls, messages: list[Message]) -> str:
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            parts.append(text)
        return "\n".join(parts)
