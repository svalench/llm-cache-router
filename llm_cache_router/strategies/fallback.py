from __future__ import annotations

import asyncio

from llm_cache_router.models import LLMResponse
from llm_cache_router.providers.base import ProviderError


class AllProvidersFailedError(RuntimeError):
    pass


class FallbackChainStrategy:
    def __init__(self, chain: list[str], timeout: float = 10.0) -> None:
        self.chain = chain
        self.timeout = timeout

    async def execute(self, call_provider) -> LLMResponse:  # noqa: ANN001
        errors: list[str] = []
        for provider_model in self.chain:
            try:
                return await asyncio.wait_for(call_provider(provider_model), timeout=self.timeout)
            except (TimeoutError, ProviderError, asyncio.TimeoutError) as exc:
                errors.append(f"{provider_model}: {exc}")
        raise AllProvidersFailedError("; ".join(errors) or "All providers failed")

