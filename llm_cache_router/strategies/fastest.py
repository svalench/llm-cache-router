from __future__ import annotations

from collections.abc import Iterable


class FastestFirstStrategy:
    def __init__(self) -> None:
        self._p50_by_provider_model: dict[str, float] = {}

    def observe(self, provider_model: str, latency_ms: int) -> None:
        prev = self._p50_by_provider_model.get(provider_model)
        if prev is None:
            self._p50_by_provider_model[provider_model] = float(latency_ms)
            return
        # Упрощённое экспоненциальное сглаживание вместо хранения полной истории.
        self._p50_by_provider_model[provider_model] = (0.8 * prev) + (0.2 * latency_ms)

    def select(self, available_provider_models: Iterable[str]) -> str:
        options = list(available_provider_models)
        if not options:
            raise ValueError("No providers available")
        return min(options, key=lambda item: self._p50_by_provider_model.get(item, 10_000.0))

