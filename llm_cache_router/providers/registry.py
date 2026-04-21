from __future__ import annotations

from typing import Type

from llm_cache_router.providers.base import LLMProvider

_REGISTRY: dict[str, Type[LLMProvider]] = {}


def register_provider(name: str, cls: Type[LLMProvider]) -> None:
    _REGISTRY[name] = cls


def get_provider_class(name: str) -> Type[LLMProvider]:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unsupported provider: '{name}'. "
            f"Available providers: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def registered_providers() -> list[str]:
    return sorted(_REGISTRY.keys())
