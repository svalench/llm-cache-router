from __future__ import annotations

import math
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

        def input_price(item: tuple[str, str]) -> float:
            # Модель без известного прайсинга (например, self-hosted через
            # openai_compatible) не должна побеждать как «бесплатная».
            # Такая цена трактуется как бесконечная — выбор падает на модель
            # с известным прайсом; если неизвестны все, min() вернёт первую.
            pricing = self._pricing.get_or_none(f"{item[0]}/{item[1]}")
            if pricing is None:
                return math.inf
            return pricing["input"]

        return min(options, key=input_price)


# Backward-compat alias — не удалять до v1.0
PRICING = get_pricing_manager().all
