from __future__ import annotations

from llm_cache_router.providers.base import ProviderConfig
from llm_cache_router.providers.openai import OpenAIProvider
from llm_cache_router.providers.registry import register_provider


class OpenAICompatibleProvider(OpenAIProvider):
    """Универсальный провайдер для любого OpenAI-compatible endpoint.

    Закрывает сразу целый класс сервисов с протоколом OpenAI Chat Completions:

    - OpenRouter (https://openrouter.ai/api/v1)
    - vLLM / llama.cpp server / Ollama с OpenAI-эндпоинтом
    - LiteLLM proxy, Together, Groq, DeepSeek, любые self-hosted инференсы

    ``base_url`` обязателен и должен указывать на корень API (обычно заканчивается
    на ``/v1``). ``api_key`` опционален — локальные серверы часто работают без него.

    Пример:

    ```python
    router = LLMRouter(
        providers={
            "openai_compatible": {
                "base_url": "http://localhost:8000/v1",  # vLLM
                "api_key": "optional-for-local-servers",
                "models": ["qwen2.5-32b-instruct"],
            }
        }
    )
    ```
    """

    _require_api_key = False

    def __init__(self, config: ProviderConfig) -> None:
        if not config.base_url:
            raise ValueError(
                "openai_compatible provider requires base_url "
                "(e.g. https://openrouter.ai/api/v1 or http://localhost:8000/v1)"
            )
        super().__init__(config)


register_provider("openai_compatible", OpenAICompatibleProvider)
