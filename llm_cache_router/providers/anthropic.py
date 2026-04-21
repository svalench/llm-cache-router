from __future__ import annotations

import json
import time
from typing import AsyncGenerator

from llm_cache_router.models import LLMResponse, LLMStreamChunk
from llm_cache_router.providers.base import LLMProvider, ProviderConfig, ProviderError
from llm_cache_router.providers.registry import register_provider
from llm_cache_router.retry import with_retry


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
        async def _call() -> LLMResponse:
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
            text_parts = [
                item.get("text", "") for item in data.get("content", []) if item.get("type") == "text"
            ]
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

        return await with_retry(
            _call,
            config=self._config.retry,
            operation_name=f"anthropic/{model}",
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 512,
            "stream": True,
        }
        input_tokens = 0
        output_tokens = 0
        final_emitted = False
        async with self._client.stream(
            "POST",
            f"{self._base_url}/messages",
            headers={
                "x-api-key": self.config.api_key or "",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise ProviderError(f"Anthropic error: {response.status_code} {body.decode(errors='ignore')}")
            event_type = ""
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if event_type == "message_start":
                    usage = data.get("message", {}).get("usage", {})
                    input_tokens = int(usage.get("input_tokens", input_tokens))
                elif event_type == "content_block_delta":
                    delta = data.get("delta", {}).get("text", "")
                    if delta:
                        yield LLMStreamChunk(
                            delta=delta,
                            provider_used="anthropic",
                            model_used=model,
                            is_final=False,
                        )
                elif event_type == "message_delta":
                    usage = data.get("usage", {})
                    output_tokens = int(usage.get("output_tokens", output_tokens))
                elif event_type == "message_stop":
                    final_emitted = True
                    yield LLMStreamChunk(
                        delta="",
                        provider_used="anthropic",
                        model_used=model,
                        is_final=True,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
        if not final_emitted:
            yield LLMStreamChunk(
                delta="",
                provider_used="anthropic",
                model_used=model,
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )


register_provider("anthropic", AnthropicProvider)

