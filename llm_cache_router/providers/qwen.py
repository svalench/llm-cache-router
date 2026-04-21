from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator

from llm_cache_router.models import LLMResponse, LLMStreamChunk
from llm_cache_router.providers.base import LLMProvider, ProviderConfig, ProviderError
from llm_cache_router.providers.registry import register_provider
from llm_cache_router.retry import with_retry


class QwenProvider(LLMProvider):
    """
    Qwen (DashScope compatible-mode) использует OpenAI-compatible chat completions API.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._base_url = config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not config.api_key:
            raise ValueError("Qwen api_key is required")

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

        return await with_retry(
            _call,
            config=self.config.retry,
            operation_name=f"qwen/{model}",
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
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        input_tokens = 0
        output_tokens = 0
        final_emitted = False
        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise ProviderError(
                    f"Qwen error: {response.status_code} {body.decode(errors='ignore')}"
                )
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                data = json.loads(raw)
                if data.get("usage"):
                    usage = data["usage"]
                    input_tokens = int(usage.get("prompt_tokens", input_tokens))
                    output_tokens = int(usage.get("completion_tokens", output_tokens))
                choices = data.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {}).get("content", "") or ""
                is_final = choice.get("finish_reason") is not None
                if is_final:
                    final_emitted = True
                yield LLMStreamChunk(
                    delta=delta,
                    provider_used="qwen",
                    model_used=model,
                    is_final=is_final,
                    input_tokens=input_tokens if is_final else None,
                    output_tokens=output_tokens if is_final else None,
                )
        if not final_emitted:
            yield LLMStreamChunk(
                delta="",
                provider_used="qwen",
                model_used=model,
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )


register_provider("qwen", QwenProvider)
