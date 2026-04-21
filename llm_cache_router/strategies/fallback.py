from __future__ import annotations

import asyncio
import logging

from llm_cache_router.models import LLMResponse

logger = logging.getLogger(__name__)


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
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fallback: %s failed: %s", provider_model, exc)
                errors.append(f"{provider_model}: {exc}")
        raise AllProvidersFailedError("; ".join(errors) or "All providers failed")
