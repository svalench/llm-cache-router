from __future__ import annotations

from collections.abc import Iterable

from llm_cache_router.pricing.manager import get_pricing_manager


class CheapestFirstStrategy:
    def __init__(self) -> None:
        self._pricing = get_pricing_manager()

    async def select(
        self,
        available_provider_models: Iterable[tuple[str, str]],
        estimated_tokens: int | None = None,
    ) -> tuple[str, str]:
        del estimated_tokens
        await self._pricing.ensure_fresh()
        options = list(available_provider_models)
        if not options:
            raise ValueError("No providers available")
        return min(
            options,
            key=lambda item: self._pricing.get(f"{item[0]}/{item[1]}").get("input", 0.0),
        )


# Backward-compat alias — не удалять до v1.0
PRICING = get_pricing_manager().all
