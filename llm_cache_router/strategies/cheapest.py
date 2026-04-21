from __future__ import annotations

from collections.abc import Iterable

# USD per 1M tokens, input/output.
PRICING: dict[str, dict[str, float]] = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-haiku": {"input": 0.25, "output": 1.25},
    "ollama/llama3.2": {"input": 0.00, "output": 0.00},
}


class CheapestFirstStrategy:
    def select(self, available_provider_models: Iterable[str], estimated_tokens: int = 0) -> str:
        del estimated_tokens
        options = list(available_provider_models)
        if not options:
            raise ValueError("No providers available")
        return min(options, key=lambda item: PRICING.get(item, {}).get("input", 999.0))
