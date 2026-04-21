from __future__ import annotations

import time

from llm_cache_router.models import LLMResponse
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
            payload = {
                "contents": [{"role": "user", "parts": [{"text": text}]}],
                "generationConfig": {"temperature": temperature},
            }
            if max_tokens is not None:
                payload["generationConfig"]["maxOutputTokens"] = max_tokens

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


register_provider("gemini", GeminiProvider)
