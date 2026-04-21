from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RoutingStrategy(StrEnum):
    CHEAPEST_FIRST = "cheapest_first"
    FASTEST_FIRST = "fastest_first"
    FALLBACK_CHAIN = "fallback_chain"


class CacheConfig(BaseModel):
    backend: str = "memory"
    threshold: float = 0.92
    ttl: int = 3600
    max_entries: int = 10_000
    min_query_length: int = 10
    embedding_model: str = "all-MiniLM-L6-v2"
    redis_url: str = "redis://localhost:6379/0"
    redis_namespace: str = "llm_cache_router"
    redis_command_timeout_sec: float = 1.5
    redis_retry_attempts: int = 3
    redis_retry_backoff_sec: float = 0.2
    redis_candidate_k: int | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "llm_cache_router"


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    content: str
    provider_used: str
    model_used: str
    cache_hit: bool = False
    cache_similarity: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    raw: dict[str, Any] | None = None


class LLMStreamChunk(BaseModel):
    delta: str
    provider_used: str
    model_used: str
    is_final: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    cache_hit: bool = False
    cache_similarity: float | None = None


class ModelUsageStat(BaseModel):
    requests: int = 0
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class WarmupEntry(BaseModel):
    messages: list[dict[str, str]]
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None


class RouterStats(BaseModel):
    cache_hit_rate: float
    total_requests: int
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    total_cost_usd: float
    saved_cost_usd: float
    provider_usage: dict[str, int] = Field(default_factory=dict)
    daily_spend_usd: float | None = None
    monthly_spend_usd: float | None = None
    budget_remaining_usd: float | None = None
    monthly_budget_remaining_usd: float | None = None
    model_usage: dict[str, ModelUsageStat] = Field(default_factory=dict)


class CacheEntry(BaseModel):
    query: str
    response: LLMResponse
    embedding: list[float]
    created_at_ts: float
    ttl: int
    hit_count: int = 0

    def is_expired(self, now_ts: float) -> bool:
        return now_ts > (self.created_at_ts + self.ttl)
