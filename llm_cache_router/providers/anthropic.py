from __future__ import annotations

import time

from llm_cache_router.models import LLMResponse
from llm_cache_router.providers.base import LLMProvider, ProviderConfig, ProviderError


class AnthropicProvider(LLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._base_url = config.base_url or "https://api.anthropic.com/v1"
        if not config.api_key:
            raise ValueError("Anthropic api_key is required")

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 512,
        }
        response = await self._client.post(
            f"{self._base_url}/messages",
            headers={
                "x-api-key": self.config.api_key or "",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        )
        if response.status_code >= 400:
            raise ProviderError(f"Anthropic error: {response.status_code} {response.text}")
        data = response.json()
        text_parts = [item.get("text", "") for item in data.get("content", []) if item.get("type") == "text"]
        usage = data.get("usage", {})
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            content="".join(text_parts),
            provider_used="anthropic",
            model_used=model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            raw=data,
        )

