from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from llm_cache_router.models import LLMResponse, LLMStreamChunk
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
            options: dict[str, Any] = {"temperature": temperature}
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": options,
            }
            if max_tokens is not None:
                options["num_predict"] = max_tokens
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
            config=self.config.retry,
            operation_name=f"ollama/{model}",
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        options: dict[str, Any] = {"temperature": temperature}
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        input_tokens = 0
        output_tokens = 0
        final_emitted = False
        async with self._client.stream(
            "POST", f"{self._base_url}/api/chat", json=payload
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise ProviderError(
                    f"Ollama error: {response.status_code} {body.decode(errors='ignore')}"
                )

            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                delta = data.get("message", {}).get("content", "") or ""
                is_final = bool(data.get("done", False))
                if is_final:
                    input_tokens = int(data.get("prompt_eval_count", 0))
                    output_tokens = int(data.get("eval_count", 0))
                    final_emitted = True
                yield LLMStreamChunk(
                    delta=delta,
                    provider_used="ollama",
                    model_used=model,
                    is_final=is_final,
                    input_tokens=input_tokens if is_final else None,
                    output_tokens=output_tokens if is_final else None,
                )

        if not final_emitted:
            yield LLMStreamChunk(
                delta="",
                provider_used="ollama",
                model_used=model,
                is_final=True,
                input_tokens=0,
                output_tokens=0,
            )


register_provider("ollama", OllamaProvider)
