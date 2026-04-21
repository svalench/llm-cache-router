from __future__ import annotations

import pytest


def test_all_builtin_providers_registered() -> None:
    import llm_cache_router.providers  # noqa: F401
    from llm_cache_router.providers.registry import registered_providers

    expected = {"openai", "anthropic", "gemini", "ollama", "minimax", "qwen"}
    assert expected.issubset(set(registered_providers()))


def test_unknown_provider_raises() -> None:
    from llm_cache_router.providers.registry import get_provider_class

    with pytest.raises(ValueError, match="Unsupported provider"):
        get_provider_class("nonexistent")


def test_custom_provider_registration() -> None:
    from llm_cache_router.models import LLMResponse
    from llm_cache_router.providers.base import LLMProvider
    from llm_cache_router.providers.registry import get_provider_class, register_provider

    class MyProvider(LLMProvider):
        async def complete(self, **kwargs):  # noqa: ANN003,ANN201
            del kwargs
            return LLMResponse(content="ok", provider_used="myprovider", model_used="m")

        async def close(self) -> None:
            return None

    register_provider("myprovider", MyProvider)
    assert get_provider_class("myprovider") is MyProvider
