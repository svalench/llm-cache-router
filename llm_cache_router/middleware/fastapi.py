from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from llm_cache_router.observability.prometheus import HTTPMetricsCollector, mount_prometheus_metrics
from llm_cache_router.router import LLMRouter

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
except ImportError:  # pragma: no cover
    BaseHTTPMiddleware = object  # type: ignore
    Request = object  # type: ignore


class LLMCacheMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """
    Middleware хранит router в request.state, чтобы его можно было использовать в хендлерах.
    """

    def __init__(self, app, router: LLMRouter) -> None:  # noqa: ANN001
        super().__init__(app)
        self._router = router

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        request.state.llm_router = self._router
        return await call_next(request)


class LLMHTTPMetricsMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """
    Снимает HTTP метрики для Prometheus: request count и latency histogram.
    """

    def __init__(self, app, collector: HTTPMetricsCollector) -> None:  # noqa: ANN001
        super().__init__(app)
        self._collector = collector

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - started
            self._collector.observe(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_seconds=duration,
            )


def cached_llm(router: LLMRouter, ttl: int | None = None) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    del ttl

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):  # noqa: ANN002,ANN003
            if "messages" in kwargs and "model" in kwargs:
                return await router.complete(kwargs["messages"], kwargs["model"])
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def mount_metrics_endpoint(  # noqa: ANN001
    app,
    router: LLMRouter,
    path: str = "/metrics",
    http_metrics: HTTPMetricsCollector | None = None,
) -> None:
    mount_prometheus_metrics(app=app, router=router, path=path, http_metrics=http_metrics)


def add_http_metrics_middleware(
    app,  # noqa: ANN001
    buckets: tuple[float, ...] | None = None,
) -> HTTPMetricsCollector:
    collector = HTTPMetricsCollector(buckets=buckets)
    app.state.llm_http_metrics = collector
    app.add_middleware(LLMHTTPMetricsMiddleware, collector=collector)
    return collector

