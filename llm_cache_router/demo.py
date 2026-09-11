"""Offline onboarding demo; responses are fixed text, not model-generated."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_cache_router import CacheConfig, LLMRouter, RoutingStrategy
from llm_cache_router.models import LLMResponse, Message
from llm_cache_router.providers.base import LLMProvider, ProviderConfig
from llm_cache_router.providers.registry import register_provider


class DemoProvider(LLMProvider):
    """A stub provider with no HTTP client, credentials, or model weights."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del messages, temperature, max_tokens
        return LLMResponse(
            content="This is a fixed demo response, not an LLM-generated answer.",
            provider_used=self.config.name,
            model_used=model,
        )

    async def close(self) -> None:
        pass


async def run_demo() -> dict[str, Any]:
    """Exercise a real router/cache miss and hit without making network requests."""
    register_provider("offline_demo", DemoProvider)
    async with LLMRouter(
        providers={"offline_demo": {"models": ["demo-model"]}},
        cache=CacheConfig(
            backend="memory",
            embedding_model="hash",  # No sentence-transformer model download.
            exact_match=True,  # Demonstrate cache mechanics, not semantic quality.
        ),
        # Unlike CHEAPEST_FIRST, this strategy does not refresh remote pricing.
        strategy=RoutingStrategy.FASTEST_FIRST,
    ) as router:
        messages: list[Message] = [{"role": "user", "content": "What is a semantic cache?"}]
        first = await router.complete(messages=messages, model="demo-model")
        second = await router.complete(messages=messages, model="demo-model")
        stats = router.stats()
        return {
            "mode": "offline stub / exact-match cache",
            "response": first.content,
            "first_cache_hit": first.cache_hit,
            "second_cache_hit": second.cache_hit,
            "provider_calls": stats.provider_usage.get("offline_demo", 0),
            "total_requests": stats.total_requests,
            "cache_hits": stats.cache_hits,
            "total_cost_usd": stats.total_cost_usd,
        }


def main() -> None:
    print(json.dumps(asyncio.run(run_demo()), indent=2))


if __name__ == "__main__":
    main()
