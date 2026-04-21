# llm-cache-router

Лёгкая Python-библиотека для:
- семантического кэширования LLM-запросов;
- multi-provider роутинга;
- трекинга стоимости и бюджетных лимитов.

## Установка

```bash
pip install llm-cache-router

# опциональные бэкенды
pip install "llm-cache-router[redis]"
pip install "llm-cache-router[qdrant]"
pip install "llm-cache-router[fastapi]"
pip install "llm-cache-router[all]"
```

## Быстрый старт — complete()

```python
from llm_cache_router import CacheConfig, LLMRouter, RoutingStrategy

router = LLMRouter(
    providers={
        "openai":    {"api_key": "sk-...",           "models": ["gpt-4o-mini"]},
        "anthropic": {"api_key": "sk-ant-...",       "models": ["claude-3-5-sonnet"]},
        "gemini":    {"api_key": "AIza...",          "models": ["gemini-1.5-flash"]},
        "minimax":   {"api_key": "minimax-...",      "models": ["MiniMax-Text-01"]},
        "qwen":      {"api_key": "dashscope-...",    "models": ["qwen-plus"]},
        "ollama":    {"base_url": "http://localhost:11434", "models": ["llama3.2"]},
    },
    cache=CacheConfig(
        backend="memory",
        threshold=0.92,
        ttl=3600,
        max_entries=10_000,
    ),
    strategy=RoutingStrategy.CHEAPEST_FIRST,
    budget={"daily_usd": 5.0, "monthly_usd": 50.0},
)

response = await router.complete(
    messages=[{"role": "user", "content": "Что такое семантический кэш?"}],
    model="gpt-4o-mini",
)
print(response.content, response.cache_hit, response.cost_usd)
```

## Стриминг — stream()

Все провайдеры (OpenAI, Anthropic, Gemini, Ollama, MiniMax, Qwen) поддерживают
нативный стриминг. Кэш работает прозрачно: при cache hit возвращается один
финальный чанк, при cache miss — реальный поток от провайдера.

```python
async for chunk in router.stream(
    messages=[{"role": "user", "content": "Объясни async/await в Python"}],
    model="gpt-4o-mini",
):
    print(chunk.delta, end="", flush=True)
    if chunk.is_final:
        print()
        print(f"provider={chunk.provider_used}, cost=${chunk.cost_usd:.6f}")
```

## Прогрев кэша — warmup()

```python
from llm_cache_router.models import WarmupEntry

results = await router.warmup(
    entries=[
        WarmupEntry(
            messages=[{"role": "user", "content": "Что такое RAG?"}],
            model="gpt-4o-mini",
        ),
        WarmupEntry(
            messages=[{"role": "user", "content": "Объясни векторные базы данных"}],
            model="gpt-4o-mini",
        ),
    ],
    concurrency=5,
    skip_cached=True,
)
print(results)  # {"warmed": 2, "skipped": 0, "failed": 0}
```

## Стратегии роутинга

| Стратегия | Описание |
|---|---|
| `CHEAPEST_FIRST` | Выбирает самый дешёвый провайдер по текущим ценам |
| `FASTEST_FIRST` | Выбирает провайдер с наименьшей накопленной latency |
| `FALLBACK_CHAIN` | Пробует провайдеры последовательно, переходит к следующему при ошибке |

```python
# FALLBACK_CHAIN
router = LLMRouter(
    providers={
        "openai":    {"api_key": "sk-...",     "models": ["gpt-4o"]},
        "anthropic": {"api_key": "sk-ant-...", "models": ["claude-3-5-sonnet"]},
    },
    strategy=RoutingStrategy.FALLBACK_CHAIN,
    fallback_chain=["openai/gpt-4o", "anthropic/claude-3-5-sonnet"],
)
```

## Redis backend

```python
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
        redis_candidate_k=256,
    ),
)
```

## Qdrant backend

```bash
pip install "llm-cache-router[qdrant]"
```

```python
router = LLMRouter(
    providers={"openai": {"api_key": "sk-...", "models": ["gpt-4o-mini"]}},
    cache=CacheConfig(
        backend="qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,           # опционально для Qdrant Cloud
        qdrant_collection="llm_cache",
        threshold=0.92,
        ttl=3600,
        max_entries=100_000,
    ),
)
```

## Бюджет и стоимость

```python
router = LLMRouter(
    providers={...},
    budget={
        "daily_usd": 5.0,
        "monthly_usd": 50.0,
    },
)

stats = router.stats()
print(stats.total_cost_usd)
print(stats.saved_cost_usd)          # сэкономлено через кэш
print(stats.daily_spend_usd)
print(stats.budget_remaining_usd)    # None если лимит не задан
print(stats.cache_hit_rate)          # 0.0–1.0
```

## FastAPI интеграция

```bash
pip install "llm-cache-router[fastapi]"
```

```python
from fastapi import FastAPI
from llm_cache_router.middleware.fastapi import (
    add_http_metrics_middleware,
    mount_metrics_endpoint,
)

app = FastAPI()
add_http_metrics_middleware(app=app)
mount_metrics_endpoint(app=app, router=router, path="/metrics")
```

Метрики Prometheus:
- `llm_router_http_requests_total{method,path,status}`
- `llm_router_http_request_duration_seconds_*`

## Async context manager

```python
async with LLMRouter(providers={...}) as router:
    response = await router.complete(messages=[...], model="gpt-4o-mini")
# close() вызывается автоматически
```

## Структура проекта

```text
llm_cache_router/
  cache/         — memory, redis, qdrant backends
  providers/     — openai, anthropic, gemini, ollama, minimax, qwen
  strategies/    — cheapest, fastest, fallback
  embeddings/    — SentenceEncoder, HashingEncoder
  cost/          — CostTracker с дневным/месячным бюджетом
  middleware/    — FastAPI middleware + Prometheus
  models.py
  router.py
```

## Тесты

```bash
pip install "llm-cache-router[dev]"
pytest
```

## Roadmap

- v0.3: Django helpers;
- v0.4: streaming retry (reconnect при обрыве SSE);
- v1.0: OpenTelemetry трейсинг.
