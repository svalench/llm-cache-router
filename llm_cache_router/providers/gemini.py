from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from llm_cache_router.models import LLMResponse, LLMStreamChunk
from llm_cache_router.providers.base import LLMProvider, ProviderConfig, ProviderError
from llm_cache_router.providers.registry import register_provider
from llm_cache_router.retry import with_retry


class GeminiProvider(LLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        if not config.api_key:
            raise ValueError("Gemini api_key is required")

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        async def _call() -> LLMResponse:
            started = time.perf_counter()
            text = self._extract_text_from_messages(messages)
            generation_config: dict[str, Any] = {"temperature": temperature}
            payload: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": text}]}],
                "generationConfig": generation_config,
            }
            if max_tokens is not None:
                generation_config["maxOutputTokens"] = max_tokens

            response = await self._client.post(
                f"{self._base_url}/models/{model}:generateContent?key={self.config.api_key}",
                json=payload,
            )
            if response.status_code >= 400:
                raise ProviderError(f"Gemini error: {response.status_code} {response.text}")
            data = response.json()
            candidate = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            latency_ms = int((time.perf_counter() - started) * 1000)
            return LLMResponse(
                content=candidate,
                provider_used="gemini",
                model_used=model,
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
                latency_ms=latency_ms,
                raw=data,
            )

        return await with_retry(
            _call,
            config=self._config.retry,
            operation_name=f"gemini/{model}",
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        text = self._extract_text_from_messages(messages)
        generation_config: dict[str, Any] = {"temperature": temperature}
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": generation_config,
        }
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        url = (
            f"{self._base_url}/models/{model}:streamGenerateContent"
            f"?alt=sse&key={self.config.api_key}"
        )
        input_tokens = 0
        output_tokens = 0
        final_emitted = False
        async with self._client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise ProviderError(
                    f"Gemini error: {response.status_code} {body.decode(errors='ignore')}"
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if not raw:
                    continue
                data = json.loads(raw)
                usage = data.get("usageMetadata", {})
                if usage:
                    input_tokens = int(usage.get("promptTokenCount", input_tokens))
                    output_tokens = int(usage.get("candidatesTokenCount", output_tokens))

                candidates = data.get("candidates", [])
                if not candidates:
                    continue
                candidate = candidates[0]
                parts = candidate.get("content", {}).get("parts", [])
                delta = "".join(part.get("text", "") for part in parts)
                finish_reason = candidate.get("finishReason")
                is_final = (
                    finish_reason is not None and finish_reason != "FINISH_REASON_UNSPECIFIED"
                )
                if is_final:
                    final_emitted = True

                yield LLMStreamChunk(
                    delta=delta,
                    provider_used="gemini",
                    model_used=model,
                    is_final=is_final,
                    input_tokens=input_tokens if is_final else None,
                    output_tokens=output_tokens if is_final else None,
                )

        if not final_emitted:
            yield LLMStreamChunk(
                delta="",
                provider_used="gemini",
                model_used=model,
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )


register_provider("gemini", GeminiProvider)
