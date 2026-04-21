from __future__ import annotations

import time

from llm_cache_router.models import LLMResponse
from llm_cache_router.providers.base import LLMProvider, ProviderConfig, ProviderError
from llm_cache_router.providers.registry import register_provider


class QwenProvider(LLMProvider):
    """
    Qwen (DashScope compatible-mode) использует OpenAI-compatible chat completions API.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._base_url = config.base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        if not config.api_key:
            raise ValueError("Qwen api_key is required")

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
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json=payload,
        )
        if response.status_code >= 400:
            raise ProviderError(f"Qwen error: {response.status_code} {response.text}")
        data = response.json()
        usage = data.get("usage", {})
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            provider_used="qwen",
            model_used=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            raw=data,
        )


register_provider("qwen", QwenProvider)

