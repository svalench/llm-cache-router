from llm_cache_router.models import (
    CacheConfig,
    LLMResponse,
    LLMStreamChunk,
    RouterStats,
    RoutingStrategy,
)
from llm_cache_router.router import LLMRouter

try:
    from llm_cache_router.middleware.fastapi import (
        add_http_metrics_middleware,
        cached_llm,
        mount_metrics_endpoint,
    )
    from llm_cache_router.observability.prometheus import (
        HTTPMetricsCollector,
        build_prometheus_metrics,
    )
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    add_http_metrics_middleware = None  # type: ignore[assignment]
    cached_llm = None  # type: ignore[assignment]
    mount_metrics_endpoint = None  # type: ignore[assignment]
    HTTPMetricsCollector = None  # type: ignore[misc,assignment]
    build_prometheus_metrics = None  # type: ignore[assignment]

__all__ = [
    "LLMRouter",
    "CacheConfig",
    "RoutingStrategy",
    "LLMResponse",
    "LLMStreamChunk",
    "RouterStats",
    "cached_llm",
    "mount_metrics_endpoint",
    "add_http_metrics_middleware",
    "build_prometheus_metrics",
    "HTTPMetricsCollector",
]
