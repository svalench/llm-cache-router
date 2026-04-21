from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from llm_cache_router.router import LLMRouter

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class HTTPMetricsCollector:
    def __init__(self, buckets: tuple[float, ...] | None = None) -> None:
        self._buckets = buckets or (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        self._requests_total: dict[tuple[str, str, int], int] = defaultdict(int)
        self._duration_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_count: dict[tuple[str, str], int] = defaultdict(int)
        self._duration_bucket: dict[tuple[str, str, float], int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, method: str, path: str, status_code: int, duration_seconds: float) -> None:
        method_v = method.upper()
        key_status = (method_v, path, int(status_code))
        key_base = (method_v, path)
        with self._lock:
            self._requests_total[key_status] += 1
            self._duration_sum[key_base] += duration_seconds
            self._duration_count[key_base] += 1
            for bucket in self._buckets:
                if duration_seconds <= bucket:
                    self._duration_bucket[(method_v, path, bucket)] += 1
            self._duration_bucket[(method_v, path, float("inf"))] += 1

    def build_metrics(self) -> str:
        lines: list[str] = []
        lines.append("# HELP llm_router_http_requests_total Total HTTP requests by method/path/status.")
        lines.append("# TYPE llm_router_http_requests_total counter")
        with self._lock:
            for (method, path, status), value in sorted(self._requests_total.items()):
                lines.append(
                    'llm_router_http_requests_total'
                    f'{{method="{method}",path="{_escape_label(path)}",status="{status}"}} {value}'
                )

            lines.append(
                "# HELP llm_router_http_request_duration_seconds HTTP request duration histogram."
            )
            lines.append("# TYPE llm_router_http_request_duration_seconds histogram")
            for method, path in sorted(self._duration_count.keys()):
                for bucket in self._buckets:
                    bucket_value = self._duration_bucket.get((method, path, bucket), 0)
                    lines.append(
                        'llm_router_http_request_duration_seconds_bucket'
                        f'{{method="{method}",path="{_escape_label(path)}",le="{bucket}"}} {bucket_value}'
                    )
                inf_value = self._duration_bucket.get((method, path, float("inf")), 0)
                lines.append(
                    'llm_router_http_request_duration_seconds_bucket'
                    f'{{method="{method}",path="{_escape_label(path)}",le="+Inf"}} {inf_value}'
                )
                lines.append(
                    'llm_router_http_request_duration_seconds_sum'
                    f'{{method="{method}",path="{_escape_label(path)}"}} '
                    f'{self._duration_sum[(method, path)]}'
                )
                lines.append(
                    'llm_router_http_request_duration_seconds_count'
                    f'{{method="{method}",path="{_escape_label(path)}"}} '
                    f'{self._duration_count[(method, path)]}'
                )
        return "\n".join(lines)


def build_prometheus_metrics(router: LLMRouter, http_metrics: HTTPMetricsCollector | None = None) -> str:
    snapshot = router.metrics_snapshot()
    lines: list[str] = []

    _append_metric(lines, "llm_router_requests_total", "counter", snapshot["total_requests"])
    _append_metric(lines, "llm_router_cache_hits_total", "counter", snapshot["cache_hits"])
    _append_metric(lines, "llm_router_cache_misses_total", "counter", snapshot["cache_misses"])
    _append_metric(lines, "llm_router_cache_hit_rate", "gauge", snapshot["cache_hit_rate"])
    _append_metric(lines, "llm_router_cache_evictions_total", "counter", snapshot["cache_evictions"])
    _append_metric(lines, "llm_router_cost_total_usd", "gauge", snapshot["total_cost_usd"])
    _append_metric(lines, "llm_router_saved_cost_total_usd", "gauge", snapshot["saved_cost_usd"])

    _append_optional_metric(lines, "llm_router_daily_spend_usd", "gauge", snapshot["daily_spend_usd"])
    _append_optional_metric(
        lines,
        "llm_router_monthly_spend_usd",
        "gauge",
        snapshot["monthly_spend_usd"],
    )
    _append_optional_metric(
        lines,
        "llm_router_budget_remaining_usd",
        "gauge",
        snapshot["budget_remaining_usd"],
    )
    _append_optional_metric(
        lines,
        "llm_router_monthly_budget_remaining_usd",
        "gauge",
        snapshot["monthly_budget_remaining_usd"],
    )

    provider_usage: dict[str, int] = snapshot["provider_usage"]
    lines.append("# HELP llm_router_provider_requests_total Total requests by provider.")
    lines.append("# TYPE llm_router_provider_requests_total counter")
    for provider, count in sorted(provider_usage.items()):
        lines.append(f'llm_router_provider_requests_total{{provider="{provider}"}} {int(count)}')

    cache_backend_stats: dict[str, Any] = snapshot["cache_backend_stats"]
    for key, value in sorted(cache_backend_stats.items()):
        metric_name = f"llm_router_cache_backend_{_sanitize(key)}"
        _append_metric(lines, metric_name, "gauge", value)

    if http_metrics is not None:
        http_payload = http_metrics.build_metrics()
        if http_payload:
            lines.append(http_payload)

    lines.append("")
    return "\n".join(lines)


def mount_prometheus_metrics(
    app: Any,
    router: LLMRouter,
    path: str = "/metrics",
    http_metrics: HTTPMetricsCollector | None = None,
) -> None:
    try:
        from fastapi import Response
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is required to mount Prometheus metrics endpoint") from exc

    @app.get(path, include_in_schema=False)  # type: ignore[misc]
    async def _metrics() -> Response:
        collector = http_metrics
        if collector is None and hasattr(app, "state") and hasattr(app.state, "llm_http_metrics"):
            collector = app.state.llm_http_metrics
        payload = build_prometheus_metrics(router, http_metrics=collector)
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


def _append_metric(lines: list[str], name: str, metric_type: str, value: Any) -> None:
    lines.append(f"# HELP {name} {name.replace('_', ' ').capitalize()}.")
    lines.append(f"# TYPE {name} {metric_type}")
    lines.append(f"{name} {_to_float(value)}")


def _append_optional_metric(lines: list[str], name: str, metric_type: str, value: Any) -> None:
    if value is None:
        return
    _append_metric(lines, name, metric_type, value)


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

