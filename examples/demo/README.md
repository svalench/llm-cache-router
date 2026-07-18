# Demo stack (Docker Compose)

Запуск полного стека за одну команду: **Redis + Qdrant + FastAPI** с семантическим кэшем.

## Быстрый старт

```bash
# из корня репозитория
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

# похожий запрос — semantic cache hit
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Explain semantic caching for LLMs"}'

# метрики роутера
curl http://localhost:8000/stats

# Prometheus
curl http://localhost:8000/metrics
```

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
