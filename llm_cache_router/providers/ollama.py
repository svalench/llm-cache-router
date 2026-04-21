from __future__ import annotations

import time

from llm_cache_router.models import LLMResponse
from llm_cache_router.providers.base import LLMProvider, ProviderConfig, ProviderError
from llm_cache_router.providers.registry import register_provider
from llm_cache_router.retry import with_retry


class OllamaProvider(LLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._base_url = config.base_url or "http://localhost:11434"

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
                "stream": False,
                "options": {"temperature": temperature},
            }
            if max_tokens is not None:
                payload["options"]["num_predict"] = max_tokens
            response = await self._client.post(f"{self._base_url}/api/chat", json=payload)
            if response.status_code >= 400:
                raise ProviderError(f"Ollama error: {response.status_code} {response.text}")
            data = response.json()
            usage_prompt = int(data.get("prompt_eval_count", 0))
            usage_completion = int(data.get("eval_count", 0))
            latency_ms = int((time.perf_counter() - started) * 1000)
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                provider_used="ollama",
                model_used=model,
                input_tokens=usage_prompt,
                output_tokens=usage_completion,
                latency_ms=latency_ms,
                raw=data,
            )

        return await with_retry(
            _call,
            config=self._config.retry,
            operation_name=f"ollama/{model}",
        )


register_provider("ollama", OllamaProvider)
