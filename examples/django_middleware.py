from __future__ import annotations

from llm_cache_router import CacheConfig, LLMRouter, RoutingStrategy

llm_router = LLMRouter(
    providers={"ollama": {"base_url": "http://localhost:11434", "models": ["llama3.2"]}},
    cache=CacheConfig(backend="memory", threshold=0.9, ttl=1800),
    strategy=RoutingStrategy.FALLBACK_CHAIN,
)


class LLMRouterMiddleware:
    """
    Пример простого Django middleware для передачи router в request.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.llm_router = llm_router
        return self.get_response(request)

