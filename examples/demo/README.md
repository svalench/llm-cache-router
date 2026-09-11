# Demo stack (Docker Compose)

Стек **Redis + Qdrant + FastAPI** с семантическим кэшем и OpenAI.

**This is a paid-API example, not the offline quickstart.** It requires Docker
Compose and an OpenAI API key, makes billable calls on cache misses, and may
download embedding-model weights. For the keyless, network-free demo after
installation, use the [offline quickstart](../../README.md#quickstart-offline-no-api-key).

## Быстрый старт

```bash
git clone https://github.com/svalench/llm-cache-router.git
cd llm-cache-router
cp .env.example .env
# отредактируйте OPENAI_API_KEY в .env

docker compose up --build
```

## Проверка

```bash
# health
curl http://localhost:8000/health

# первый запрос — cache miss
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is a semantic cache?"}'

# тот же запрос — ожидается cache hit (до истечения TTL)
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is a semantic cache?"}'

# метрики роутера
curl http://localhost:8000/stats

# Prometheus
curl http://localhost:8000/metrics
```

Для похожего, но не идентичного запроса cache hit не гарантируется:
результат зависит от encoder и порога similarity. Лимит бюджета проверяется
после вызова провайдера и не является жёстким ограничением расходов.

## Qdrant backend

В `.env` установите:

```env
CACHE_BACKEND=qdrant
```

Перезапустите: `docker compose up --build`.

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `OPENAI_API_KEY` | — | Обязательный ключ OpenAI |
| `CACHE_BACKEND` | `redis` | `redis` или `qdrant` |
| `CACHE_THRESHOLD` | `0.92` | Порог cosine similarity |
| `CACHE_TTL` | `3600` | TTL кэша (сек) |
| `LLM_MODEL` | `gpt-4o-mini` | Модель OpenAI |
| `DAILY_BUDGET_USD` | `5.0` | Дневной бюджет |
