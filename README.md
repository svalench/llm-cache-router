# llm-cache-router

[![PyPI version](https://badge.fury.io/py/llm-cache-router.svg)](https://pypi.org/project/llm-cache-router/)
[![Python versions](https://img.shields.io/pypi/pyversions/llm-cache-router.svg)](https://pypi.org/project/llm-cache-router/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/llm-cache-router.svg)](https://pypi.org/project/llm-cache-router/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/svalench/llm-cache-router/actions/workflows/ci.yml/badge.svg)](https://github.com/svalench/llm-cache-router/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/svalench/llm-cache-router/blob/main/notebooks/playground.ipynb)

> A Python library that combines **semantic caching**, **multi-provider LLM routing**, and **cost tracking** in a single async-first API. Start with the offline demo below to try the cache without an API key.

---

## Table of Contents

- [Why llm-cache-router](#why-llm-cache-router)
- [Scope and limitations](#scope-and-limitations)
- [Features](#features)
- [Installation](#installation)
- [Quickstart (offline, no API key)](#quickstart-offline-no-api-key)
- [Docker demo (OpenAI API key required)](#docker-demo-openai-api-key-required)
- [Interactive Playground (Colab)](#interactive-playground-colab)
- [Connect a real provider](#connect-a-real-provider)
- [Streaming](#streaming)
- [Cache Warmup](#cache-warmup)
- [Routing Strategies](#routing-strategies)
- [Cache Backends](#cache-backends)
- [Cache Invalidation, Versioning & Exact Match](#cache-invalidation-versioning--exact-match)
- [Multimodal Messages & Cache Keys](#multimodal-messages--cache-keys)
- [Budget and Cost Tracking](#budget-and-cost-tracking)
- [FastAPI Integration](#fastapi-integration)
- [Async Context Manager](#async-context-manager)
- [Supported Providers](#supported-providers)
- [Architecture](#architecture)
- [Development](#development)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why llm-cache-router

Add caching, routing, and usage accounting without running a separate proxy:

- **Reuse responses** — a cache hit avoids another provider call. Savings depend on query repetition, cache settings, model pricing, and acceptable answer reuse; this project does not provide a measured production savings benchmark.
- **Handle provider failures** — configure fallback chains across providers and models.
- **Track cost** — per-model estimates, daily/monthly budget accounting, and Prometheus metrics.

One async API. Six named providers plus OpenAI-compatible endpoints. Three cache backends.

## Scope and limitations

- The package is **beta**. Validate cache correctness, latency, and savings on your own workload before production use.
- Semantic similarity is not a correctness guarantee. The offline demo uses exact matching and a hash encoder; it does not demonstrate semantic understanding or benchmark savings.
- Budget counters are process-local and checked **after** provider usage is recorded, not before a billable request. They are not a hard provider-side spending cap.
- The default embedding model may download weights on first use. If loading fails, the current cache backends fall back to hashing; hash similarity is not a substitute for semantic embeddings.

## Features

- **Semantic cache** — vector-similarity matching via `sentence-transformers`, not just exact string hashing. Optional `exact_match` mode, key versioning and manual invalidation for correctness-sensitive workloads.
- **Multimodal-aware cache keys** — images, audio, and video blocks are hashed into the query; cache is scoped per requested `model`.
- **Multi-provider routing** across OpenAI, Anthropic, Google Gemini, Ollama, MiniMax, Qwen (Dashscope) and any OpenAI-compatible endpoint (OpenRouter, vLLM, llama.cpp server, LiteLLM proxy, self-hosted inference).
- **Three routing strategies**: `CHEAPEST_FIRST`, `FASTEST_FIRST`, `FALLBACK_CHAIN`.
- **Pluggable cache backends**: in-memory (FAISS), Redis, Qdrant.
- **Streaming** — native async SSE streaming for every provider, transparent to the cache layer.
- **Cost tracker** with per-model pricing, daily/monthly budget limits and savings accounting.
- **Cache warmup** with controlled concurrency for pre-production pre-loading.
- **FastAPI middleware** + Prometheus metrics endpoint out of the box.
- **Typed** — Pydantic v2 models everywhere, fully typed public API.
- **Tested** — unit tests covering router, cache (incl. multimodal keys, model isolation, invalidation, key versioning), strategies, embeddings, providers, retry, warmup, HTTP middleware, and the offline demo.

> **Latest:** [v0.3.1 release notes](docs/releases/v0.3.1.md): offline demo, corrected onboarding, and distribution validation.

## Installation

```bash
pip install llm-cache-router
```

Optional extras:

```bash
pip install "llm-cache-router[redis]"     # Redis cache backend
pip install "llm-cache-router[qdrant]"    # Qdrant vector cache backend
pip install "llm-cache-router[fastapi]"   # FastAPI middleware + Prometheus
pip install "llm-cache-router[all]"       # everything above
pip install "llm-cache-router[dev]"       # tests, ruff, mypy
```

Requires **Python 3.11+**.

The core install includes **Pydantic, HTTPX, NumPy, FAISS CPU, and sentence-transformers** (which also brings in PyTorch and other dependencies). It is not a single-dependency or small-download install. Package installation needs network access unless those packages are already available locally.

## Quickstart (offline, no API key)

Use a checkout to run the demo from this branch, including changes not yet released to PyPI:

```bash
git clone https://github.com/svalench/llm-cache-router.git
cd llm-cache-router
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
llm-cache-router demo
```

After installation, the demo runs **without network requests, API keys, model downloads, Docker, Redis, or Qdrant**. It uses a fixed stub response, an in-memory cache, `embedding_model="hash"`, and `exact_match=True`. `FASTEST_FIRST` avoids the remote pricing refresh used by `CHEAPEST_FIRST`.

Expected output:

```json
{
  "mode": "offline stub / exact-match cache",
  "response": "This is a fixed demo response, not an LLM-generated answer.",
  "first_cache_hit": false,
  "second_cache_hit": true,
  "provider_calls": 1,
  "total_requests": 2,
  "cache_hits": 1,
  "total_cost_usd": 0.0
}
```

The second **identical** request reuses the first response. Each run starts with an empty cache. The zero cost is a property of this stub, not a savings estimate for real models. The demo is included in the installed package; `python -m llm_cache_router.demo` is an equivalent invocation. See [the demo implementation](llm_cache_router/demo.py) for a small custom-provider example.

## Docker demo (OpenAI API key required)

Full demo stack: **Redis + Qdrant + FastAPI** with semantic cache.
This separate example makes billable OpenAI calls on cache misses and may download embedding-model weights. It requires Docker Compose.

```bash
git clone https://github.com/svalench/llm-cache-router.git
cd llm-cache-router
cp .env.example .env
# Edit .env and set OPENAI_API_KEY before starting.
docker compose up --build
```

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is a semantic cache?"}'
```

Switch cache backend in `.env`: `CACHE_BACKEND=redis` (default) or `CACHE_BACKEND=qdrant`.

Details: [examples/demo/README.md](examples/demo/README.md).

## Interactive Playground (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/svalench/llm-cache-router/blob/main/notebooks/playground.ipynb)

Notebook: [notebooks/playground.ipynb](notebooks/playground.ipynb) — installation, caching, streaming, and `router.stats()` with the in-memory backend (no Redis/Qdrant required in Colab). It requires an OpenAI API key, can incur API charges, and may download embedding weights. Similar queries are not guaranteed cache hits.

## Connect a real provider

This example requires `OPENAI_API_KEY` in your environment and makes a billable request on a cache miss. Unlike the offline demo, it uses the default sentence-transformer encoder and may download weights.

```python
import asyncio
import os

from llm_cache_router import CacheConfig, LLMRouter


async def main() -> None:
    async with LLMRouter(
        providers={
            "openai": {
                "api_key": os.environ["OPENAI_API_KEY"],
                "models": ["gpt-4o-mini"],
            },
        },
        cache=CacheConfig(backend="memory", threshold=0.92, ttl=3600),
    ) as router:
        response = await router.complete(
            messages=[{"role": "user", "content": "What is a semantic cache?"}],
            model="gpt-4o-mini",
        )
        print(response.content)
        print(f"cache_hit={response.cache_hit}")
        print(f"total_cost_usd={router.stats().total_cost_usd:.6f}")


asyncio.run(main())
```

## Streaming

All providers (OpenAI, Anthropic, Gemini, Ollama, MiniMax, Qwen) support native SSE streaming. The cache layer is transparent: on a cache hit you receive a single final chunk, on a miss — a real streaming response that is also written to the cache once complete.

```python
async for chunk in router.stream(
    messages=[{"role": "user", "content": "Explain async/await in Python"}],
    model="gpt-4o-mini",
):
    print(chunk.delta, end="", flush=True)
    if chunk.is_final:
        print(f"\nprovider={chunk.provider_used} cost=${chunk.cost_usd:.6f}")
```

## Cache Warmup

Pre-load the cache with known queries before traffic hits production:

```python
from llm_cache_router.models import WarmupEntry

results = await router.warmup(
    entries=[
        WarmupEntry(
            messages=[{"role": "user", "content": "What is RAG?"}],
            model="gpt-4o-mini",
        ),
        WarmupEntry(
            messages=[{"role": "user", "content": "Explain vector databases"}],
            model="gpt-4o-mini",
        ),
    ],
    concurrency=5,
    skip_cached=True,
)
print(results)  # {"warmed": 2, "skipped": 0, "failed": 0}
```

## Routing Strategies

| Strategy | Description |
|---|---|
| `CHEAPEST_FIRST` | Picks the cheapest provider/model by live pricing for each call. |
| `FASTEST_FIRST` | Picks the provider with the lowest observed latency (EMA). |
| `FALLBACK_CHAIN` | Tries providers in order, falls back on error/timeout. |

```python
router = LLMRouter(
    providers={
        "openai":    {"api_key": "sk-...",     "models": ["gpt-4o"]},
        "anthropic": {"api_key": "sk-ant-...", "models": ["claude-3-5-sonnet"]},
    },
    strategy=RoutingStrategy.FALLBACK_CHAIN,
    fallback_chain=["openai/gpt-4o", "anthropic/claude-3-5-sonnet"],
)
```

## Cache Backends

### In-memory (FAISS)

Default. Zero dependencies beyond the core install. Best for single-process apps and tests.

```python
cache=CacheConfig(backend="memory", threshold=0.92, ttl=3600, max_entries=10_000)
```

### Redis

Production-grade distributed cache with LRU eviction, configurable timeouts, retry/backoff and bounded candidate set for vector search.

```python
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
)
```

### Qdrant

Native vector database for very large caches (millions of entries) and cross-service deployments.

```bash
pip install "llm-cache-router[qdrant]"
```

```python
cache=CacheConfig(
    backend="qdrant",
    qdrant_url="http://localhost:6333",
    qdrant_api_key=None,           # optional for Qdrant Cloud
    qdrant_collection="llm_cache",
    threshold=0.92,
    ttl=3600,
    max_entries=100_000,
)
```

## Cache Invalidation, Versioning & Exact Match

A semantic cache returns answers for *similar* queries — which also means it can serve stale or simply wrong answers. Three tools keep this under control:

### Choosing a safe threshold

The default `threshold=0.92` is deliberately strict, but cosine similarity cannot fully distinguish «how do I reset my password?» from «how do I reset my **admin** password?». If false positives are expensive in your domain:

- raise the threshold (0.95+), or
- enable `exact_match=True` — the cache then requires identical extracted query text within the requested model/key-version scope (not byte-identical raw message objects), or
- treat the cache as an advisory layer and validate downstream.

### Manual invalidation

```python
# Drop cached entries for one model (e.g. after a prompt/model update)
removed = await router.invalidate_cache(model="gpt-4o-mini")

# Drop everything
removed = await router.invalidate_cache()

# Full clear (all models, all key versions)
await router.clear_cache()
```

### Key versioning

Deployed a new system prompt? Bump `key_version` instead of flushing: old entries become invisible immediately and age out via TTL, while the new version starts with a clean slate.

```python
from llm_cache_router.models import CacheConfig

cache = CacheConfig(key_version="2026-09-04-prompt-v2")
```

Works identically across memory, Redis and Qdrant backends.

## Multimodal Messages & Cache Keys

Messages follow the OpenAI-compatible shape: `content` can be a **string** or a **list of blocks** (`text`, `image_url`, Anthropic `image`, audio, video). The router passes your requested `model` into the cache layer so different models never share a hit for the same text.

```python
from llm_cache_router.models import Message, WarmupEntry

messages: list[Message] = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,..."},
            },
        ],
    }
]

response = await router.complete(messages=messages, model="gpt-4o-mini")
# Second call with the same text + same image → cache_hit=True
# Same text but a different image → cache miss
# Same messages but model="gpt-4o" → cache miss (different model scope)
```

Warmup supports the same multimodal payloads:

```python
WarmupEntry(
    messages=messages,
    model="gpt-4o-mini",
)
```

Binary media is stored in the cache key as a short `sha256` fingerprint, not the full base64 payload.

## Budget and Cost Tracking

Set per-day and per-month USD limits. A `BudgetExceededError` is raised when recorded usage crosses a limit, **after the provider call has already happened**. Counters reset when the process restarts and are not shared across workers. Use provider-side limits for a hard spending cap.

Costs and savings are estimates based on reported token usage and the pricing catalog, not reconciled invoices. Unknown model prices currently contribute zero to cost accounting.

```python
router = LLMRouter(
    providers={...},
    budget={"daily_usd": 5.0, "monthly_usd": 50.0},
)

stats = router.stats()
print(stats.total_cost_usd)           # total spent since start
print(stats.saved_cost_usd)           # saved via cache hits
print(stats.daily_spend_usd)
print(stats.budget_remaining_usd)     # None if no limit is set
print(stats.cache_hit_rate)           # 0.0–1.0
```

## FastAPI Integration

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

Exposed Prometheus metrics:

- `llm_router_http_requests_total{method,path,status}`
- `llm_router_http_request_duration_seconds_*` (histogram)
- `llm_router_cache_hits_total`, `llm_router_cache_misses_total`
- `llm_router_cost_usd_total`, `llm_router_saved_cost_usd_total`

## Async Context Manager

```python
async with LLMRouter(providers={...}) as router:
    response = await router.complete(messages=[...], model="gpt-4o-mini")
# close() is called automatically — closes provider clients and cache connections
```

## Supported Providers

| Provider | Streaming | Notes |
|---|---|---|
| OpenAI | yes | `gpt-4o`, `gpt-4o-mini`, `o1-*`, etc. |
| OpenAI-compatible | yes | Any endpoint speaking the OpenAI Chat Completions protocol: OpenRouter, vLLM, llama.cpp server, LiteLLM proxy, self-hosted inference |
| Anthropic | yes | Claude 3.5 Sonnet/Haiku, Opus |
| Google Gemini | yes | 1.5 Flash, 1.5 Pro |
| Ollama | yes | Any locally-served model |
| MiniMax | yes | `MiniMax-Text-01` and others |
| Qwen (Dashscope) | yes | `qwen-plus`, `qwen-max`, etc. |

Any OpenAI-compatible endpoint works with a single provider entry — `base_url` is required, `api_key` is optional (local servers often run without one):

```python
router = LLMRouter(
    providers={
        "openai_compatible": {
            "base_url": "http://localhost:8000/v1",  # vLLM / llama.cpp / LiteLLM proxy
            "api_key": "optional",                   # omit for keyless local servers
            "models": ["qwen2.5-32b-instruct"],
        }
    }
)
```

Adding a new provider = subclass `LLMProvider`, then call `register_provider("name", YourProvider)`. See `llm_cache_router/providers/base.py` and the [offline demo](llm_cache_router/demo.py).

## Architecture

```text
llm_cache_router/
  cache/          # memory (FAISS) / redis / qdrant backends
  providers/      # openai, anthropic, gemini, ollama, minimax, qwen
  strategies/     # cheapest, fastest, fallback
  embeddings/     # SentenceEncoder, HashingEncoder
  cost/           # CostTracker with daily/monthly budgets
  middleware/     # FastAPI middleware
  observability/  # Prometheus metrics
  models.py       # Pydantic models (Message, LLMResponse, CacheEntry, ...)
  router.py       # LLMRouter — public entrypoint
  retry.py        # RetryConfig + exponential backoff
  warmup.py       # async warmup helper
```

## Development

```bash
git clone https://github.com/svalench/llm-cache-router.git
cd llm-cache-router

# using uv (recommended)
uv sync --all-extras
uv run pytest

# or plain pip
pip install -e ".[all,dev]"
pytest
```

Code quality is enforced in CI via:

- `ruff check` (lint) and `ruff format --check` (style)
- `mypy --ignore-missing-imports` (type check)
- `pytest` on Python 3.11, 3.12, 3.13 with coverage
- sdist/wheel builds, `twine check --strict`, and an installed-wheel offline smoke check

Tests use fake providers/clients and block outbound connections. They do not validate live provider APIs, semantic-model quality, or real Redis/Qdrant services. For a smaller test setup without PyTorch/model dependencies, see [CONTRIBUTING.md](CONTRIBUTING.md#lightweight-offline-unit-tests).

## Roadmap

- **v0.3** — Request tracing hooks (OpenTelemetry spans).
- **v0.4** — Streaming retry (reconnect on SSE drop); Django helpers and middleware.
- **v0.5** — Persistent budget counters (Redis/SQLite) surviving process restarts; shared EMA latency metrics for multi-worker deployments.
- **v1.0** — LLM-verified cache hits (cheap re-check of semantic matches on a mini model); pluggable pricing providers.

## Contributing

Pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Quick version:

1. Open an issue first for anything larger than a small bug fix.
2. Add tests for new behaviour.
3. Run `ruff check`, `ruff format`, `mypy` and `pytest` before pushing.

## License

MIT — see [LICENSE](LICENSE) for details.

---

## 🇷🇺 Краткое описание (Russian)

**llm-cache-router** — Python-библиотека в статусе beta для семантического кэширования LLM-запросов, мульти-провайдер роутинга и учёта стоимости. Кэш позволяет повторно использовать ответы без вызова провайдера; экономия зависит от нагрузки и настроек, подтверждённого production-бенчмарка здесь нет. Поддерживаются OpenAI, Anthropic, Gemini, Ollama, MiniMax, Qwen и OpenAI-compatible endpoints, три бэкенда кэша (in-memory / Redis / Qdrant), инвалидация и версионирование ключей, точное совпадение, стриминг и FastAPI-middleware с Prometheus-метриками. Дневные/месячные лимиты проверяются после вызова провайдера; счётчики локальны для процесса и не являются жёстким ограничением расходов.

**v0.3.0:** провайдер `openai_compatible` (OpenRouter, vLLM, llama.cpp, LiteLLM proxy), инвалидация и версионирование ключей кэша, режим точного совпадения. [Release notes](docs/releases/v0.3.0.md).

**Установка:**

```bash
pip install llm-cache-router

# с дополнительными бэкендами
pip install "llm-cache-router[redis]"
pip install "llm-cache-router[qdrant]"
pip install "llm-cache-router[fastapi]"
pip install "llm-cache-router[all]"
```

Требуется **Python 3.11+**. Полная документация и примеры — выше (на английском).

**Демо без API-ключа:** [offline quickstart](#quickstart-offline-no-api-key) — после установки `llm-cache-router demo` использует фиксированный ответ, hash encoder и точное совпадение без сетевых запросов и загрузки моделей. Установка пакета включает ML-зависимости. Docker-демо и [Colab playground](notebooks/playground.ipynb) требуют OpenAI API key и могут приводить к платным вызовам.
