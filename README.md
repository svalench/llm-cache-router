# llm-cache-router

Лёгкая Python-библиотека для:
- семантического кэширования LLM-запросов;
- multi-provider роутинга;
- трекинга стоимости и бюджетных лимитов.

## Быстрый старт

```python
from llm_cache_router import CacheConfig, LLMRouter, RoutingStrategy

router = LLMRouter(
    providers={
        "openai": {"api_key": "sk-...", "models": ["gpt-4o-mini"]},
        "anthropic": {"api_key": "sk-ant-...", "models": ["claude-3-5-sonnet"]},
        "minimax": {"api_key": "minimax-...", "models": ["MiniMax-Text-01"]},
        "qwen": {"api_key": "dashscope-...", "models": ["qwen-plus"]},
        "ollama": {"base_url": "http://localhost:11434", "models": ["llama3.2"]},
    },
    cache=CacheConfig(
        backend="memory",
        threshold=0.92,
        ttl=3600,
        max_entries=10_000,
    ),
    strategy=RoutingStrategy.CHEAPEST_FIRST,
    budget={"daily_usd": 5.0},
)

response = await router.complete(
    messages=[{"role": "user", "content": "Что такое семантический кэш?"}],
    model="gpt-4o-mini",
)
print(response.content, response.cache_hit, response.cost_usd)
```

## Что уже реализовано (v0.2)

- `LLMRouter` (async-first orchestration);
- `InMemorySemanticCache`:
  - cosine similarity;
  - `normalize_embeddings` через sentence-transformers;
  - min-length guard (`min_query_length`, по умолчанию 10);
- стратегии:
  - `CHEAPEST_FIRST`;
  - `FASTEST_FIRST` (с накоплением latency);
  - `FALLBACK_CHAIN`;
- провайдеры:
  - OpenAI;
  - Anthropic;
  - Gemini;
  - MiniMax;
  - Qwen;
  - Ollama;
- `CostTracker` с дневным и месячным бюджетом;
- `RedisSemanticCache`:
  - хранение в Redis (`redis://...`);
  - семантический top-1 поиск по cosine similarity;
  - опциональное ограничение кандидатов (`redis_candidate_k`) для ускоренного k-NN scan;
  - TTL и автоматическая очистка устаревших ссылок;
  - ограничение `max_entries` с вытеснением самых старых;
  - retry/backoff/timeout на Redis-команды;
- FastAPI интеграция:
  - `LLMCacheMiddleware`;
  - `cached_llm` decorator;
  - Prometheus endpoint (`/metrics`).

## Redis backend

```python
from llm_cache_router import CacheConfig, LLMRouter

router = LLMRouter(
    providers={"ollama": {"base_url": "http://localhost:11434", "models": ["llama3.2"]}},
    cache=CacheConfig(
        backend="redis",
        redis_url="redis://localhost:6379/0",
        redis_namespace="llm_cache_router_prod",
        threshold=0.92,
        ttl=3600,
        max_entries=50_000,
        redis_command_timeout_sec=1.5,
        redis_retry_attempts=3,
        redis_retry_backoff_sec=0.2,
        redis_candidate_k=256,  # optional
    ),
)
```

`router.stats()` теперь возвращает расширенные счётчики кэша:
- `cache_hits`
- `cache_misses`
- `cache_evictions`

Prometheus-экспорт:

```python
from llm_cache_router.middleware.fastapi import mount_metrics_endpoint
from llm_cache_router.middleware.fastapi import add_http_metrics_middleware

add_http_metrics_middleware(app=app)
mount_metrics_endpoint(app=app, router=router, path="/metrics")
```

или вручную:

```python
from llm_cache_router import build_prometheus_metrics

payload = build_prometheus_metrics(router)
```

HTTP middleware метрики:
- `llm_router_http_requests_total{method,path,status}`
- `llm_router_http_request_duration_seconds_*` (histogram buckets/sum/count)

## Структура

```text
llm_cache_router/
  cache/
  providers/
  strategies/
  embeddings/
  cost/
  middleware/
  models.py
  router.py
```

## Тесты

```bash
pytest
```

## Roadmap

- v0.3: расширенная FastAPI интеграция;
- v0.4: стабилизация Gemini/Ollama;
- v0.5: Qdrant backend + Django helpers;
- v1.0: метрики/observability.
