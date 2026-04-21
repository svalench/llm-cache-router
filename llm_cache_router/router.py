from __future__ import annotations

from typing import Any, AsyncGenerator

from llm_cache_router.cache.base import CacheBackend
from llm_cache_router.cache.memory import InMemorySemanticCache
from llm_cache_router.cache.qdrant import QdrantSemanticCache
from llm_cache_router.cache.redis import RedisSemanticCache
from llm_cache_router.cost.tracker import CostTracker
from llm_cache_router.models import (
    CacheConfig,
    LLMResponse,
    LLMStreamChunk,
    ModelUsageStat,
    RouterStats,
    RoutingStrategy,
    TokenUsage,
)
import llm_cache_router.providers  # noqa: F401
from llm_cache_router.providers.base import LLMProvider, ProviderConfig
from llm_cache_router.providers.registry import get_provider_class
from llm_cache_router.strategies.cheapest import CheapestFirstStrategy
from llm_cache_router.strategies.fallback import FallbackChainStrategy
from llm_cache_router.strategies.fastest import FastestFirstStrategy

ProviderSettings = dict[str, dict[str, Any]]


class LLMRouter:
    def __init__(
        self,
        providers: ProviderSettings,
        cache: CacheConfig | None = None,
        strategy: RoutingStrategy = RoutingStrategy.CHEAPEST_FIRST,
        budget: dict | None = None,
        fallback_chain: list[str] | None = None,
    ) -> None:
        self._provider_settings = providers
        self._providers = self._build_providers(providers)
        self._model_to_provider_names = self._build_model_index(providers)
        self._cache: CacheBackend = self._build_cache(cache or CacheConfig())
        self._cost_tracker = CostTracker(budget=budget)

        self._strategy_type = strategy
        self._cheapest = CheapestFirstStrategy()
        self._fastest = FastestFirstStrategy()
        self._fallback = FallbackChainStrategy(
            chain=fallback_chain or self._provider_model_keys(),
            timeout=10.0,
        )

        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_cost_usd = 0.0
        self._saved_cost_usd = 0.0
        self._provider_usage: dict[str, int] = {}
        self._model_usage: dict[str, ModelUsageStat] = {}

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self._total_requests += 1

        cache_entry, similarity = await self._cache.get(messages)
        if cache_entry is not None:
            self._cache_hits += 1
            self._saved_cost_usd += cache_entry.response.cost_usd
            cached = cache_entry.response.model_copy(deep=True)
            cached.cache_hit = True
            cached.cache_similarity = similarity
            return cached
        self._cache_misses += 1

        provider_model_key = self._select_provider_model(model=model, messages=messages)
        if self._strategy_type == RoutingStrategy.FALLBACK_CHAIN:
            response = await self._fallback.execute(
                lambda pm: self._call_provider(
                    provider_model_key=pm,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
        else:
            response = await self._call_provider(
                provider_model_key=provider_model_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        response.cache_hit = False
        usage = TokenUsage(input_tokens=response.input_tokens, output_tokens=response.output_tokens)
        provider_name, model_name = self._split_provider_model(response.provider_used, response.model_used)
        response.cost_usd = self._cost_tracker.record(provider_name, model_name, usage)
        self._total_cost_usd += response.cost_usd
        self._record_usage(provider_name, model_name, response)

        if self._strategy_type == RoutingStrategy.FASTEST_FIRST:
            self._fastest.observe(f"{provider_name}/{model_name}", response.latency_ms)

        await self._cache.set(messages, response)
        return response

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        self._total_requests += 1

        cache_entry, similarity = await self._cache.get(messages)
        if cache_entry is not None:
            self._cache_hits += 1
            self._saved_cost_usd += cache_entry.response.cost_usd
            yield LLMStreamChunk(
                delta=cache_entry.response.content,
                provider_used=cache_entry.response.provider_used,
                model_used=cache_entry.response.model_used,
                is_final=True,
                cache_hit=True,
                cache_similarity=similarity,
                cost_usd=0.0,
            )
            return
        self._cache_misses += 1

        provider_model_key = self._select_provider_model(model=model, messages=messages)
        provider_name, model_name = self._split_provider_model(provider_model_key, fallback_model=model)
        provider = self._providers[provider_name]

        full_content = ""
        final_recorded = False
        async for chunk in provider.stream(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            chunk.provider_used = provider_name
            chunk.model_used = model_name
            full_content += chunk.delta

            if chunk.is_final and not final_recorded:
                full_response = LLMResponse(
                    content=full_content,
                    provider_used=provider_name,
                    model_used=model_name,
                    cache_hit=False,
                    input_tokens=chunk.input_tokens or 0,
                    output_tokens=chunk.output_tokens or 0,
                )
                usage = TokenUsage(
                    input_tokens=full_response.input_tokens,
                    output_tokens=full_response.output_tokens,
                )
                full_response.cost_usd = self._cost_tracker.record(provider_name, model_name, usage)
                self._total_cost_usd += full_response.cost_usd
                self._record_usage(provider_name, model_name, full_response)
                chunk.cost_usd = full_response.cost_usd
                await self._cache.set(messages, full_response)
                final_recorded = True

            yield chunk

        if not final_recorded:
            full_response = LLMResponse(
                content=full_content,
                provider_used=provider_name,
                model_used=model_name,
                cache_hit=False,
                input_tokens=0,
                output_tokens=0,
            )
            self._record_usage(provider_name, model_name, full_response)
            await self._cache.set(messages, full_response)
            yield LLMStreamChunk(
                delta="",
                provider_used=provider_name,
                model_used=model_name,
                is_final=True,
                cost_usd=full_response.cost_usd,
            )

    def stats(self) -> RouterStats:
        total = self._total_requests or 1
        cost_stats = self._cost_tracker.stats()
        cache_stats = self._cache.stats()
        return RouterStats(
            cache_hit_rate=round(self._cache_hits / total, 4),
            total_requests=self._total_requests,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            cache_evictions=cache_stats.get("evictions", 0),
            total_cost_usd=round(self._total_cost_usd, 6),
            saved_cost_usd=round(self._saved_cost_usd, 6),
            provider_usage=dict(self._provider_usage),
            model_usage=dict(self._model_usage),
            daily_spend_usd=cost_stats["daily_spend_usd"],
            monthly_spend_usd=cost_stats["monthly_spend_usd"],
            budget_remaining_usd=cost_stats["budget_remaining_usd"],
            monthly_budget_remaining_usd=cost_stats["monthly_budget_remaining_usd"],
        )

    def metrics_snapshot(self) -> dict[str, Any]:
        router_stats = self.stats()
        cost_stats = self._cost_tracker.stats()
        cache_stats = self._cache.stats()
        return {
            "total_requests": router_stats.total_requests,
            "cache_hits": router_stats.cache_hits,
            "cache_misses": router_stats.cache_misses,
            "cache_hit_rate": router_stats.cache_hit_rate,
            "cache_evictions": router_stats.cache_evictions,
            "total_cost_usd": router_stats.total_cost_usd,
            "saved_cost_usd": router_stats.saved_cost_usd,
            "daily_spend_usd": cost_stats["daily_spend_usd"],
            "monthly_spend_usd": cost_stats["monthly_spend_usd"],
            "budget_remaining_usd": cost_stats["budget_remaining_usd"],
            "monthly_budget_remaining_usd": cost_stats["monthly_budget_remaining_usd"],
            "provider_usage": dict(self._provider_usage),
            "model_usage": {k: v.model_dump() for k, v in self._model_usage.items()},
            "cache_backend_stats": cache_stats,
        }

    async def close(self) -> None:
        if hasattr(self._cache, "close"):
            await self._cache.close()
        for provider in self._providers.values():
            await provider.close()

    def _build_cache(self, config: CacheConfig) -> CacheBackend:
        if config.backend == "memory":
            return InMemorySemanticCache(config)
        if config.backend == "redis":
            return RedisSemanticCache(config)
        if config.backend == "qdrant":
            try:
                return QdrantSemanticCache(config)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to initialize qdrant cache backend. "
                    "Install optional dependency: pip install 'llm-cache-router[qdrant]'"
                ) from exc
        raise ValueError(f"Unsupported cache backend: {config.backend}")

    @staticmethod
    def _build_providers(providers: ProviderSettings) -> dict[str, LLMProvider]:
        instances: dict[str, LLMProvider] = {}
        for provider_name, raw_cfg in providers.items():
            cfg = ProviderConfig(
                name=provider_name,
                api_key=raw_cfg.get("api_key"),
                base_url=raw_cfg.get("base_url"),
                timeout=float(raw_cfg.get("timeout", 30.0)),
            )
            provider_cls = get_provider_class(provider_name)
            instances[provider_name] = provider_cls(cfg)
        return instances

    @staticmethod
    def _build_model_index(providers: ProviderSettings) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for provider_name, cfg in providers.items():
            for model_name in cfg.get("models", []):
                index.setdefault(model_name, []).append(provider_name)
        return index

    def _provider_model_keys(self) -> list[str]:
        keys: list[str] = []
        for provider_name, cfg in self._provider_settings.items():
            for model_name in cfg.get("models", []):
                keys.append(f"{provider_name}/{model_name}")
        return keys

    def _select_provider_model(self, model: str, messages: list[dict[str, str]]) -> str:
        del messages
        providers_for_model = self._model_to_provider_names.get(model, [])
        if not providers_for_model:
            # model может приходить в формате provider/model
            if "/" in model:
                return model
            raise ValueError(f"No providers configured for model: {model}")

        provider_models = [f"{name}/{model}" for name in providers_for_model]
        if self._strategy_type == RoutingStrategy.CHEAPEST_FIRST:
            return self._cheapest.select(provider_models)
        if self._strategy_type == RoutingStrategy.FASTEST_FIRST:
            return self._fastest.select(provider_models)
        return provider_models[0]

    async def _call_provider(
        self,
        provider_model_key: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        provider_name, model_name = self._split_provider_model(provider_model_key, fallback_model=model)
        provider = self._providers[provider_name]
        response = await provider.complete(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response.provider_used = provider_name
        response.model_used = model_name
        return response

    def _record_usage(self, provider_name: str, model_name: str, response: LLMResponse) -> None:
        key = f"{provider_name}/{model_name}"
        self._provider_usage[provider_name] = self._provider_usage.get(provider_name, 0) + 1
        stat = self._model_usage.setdefault(key, ModelUsageStat())
        stat.requests += 1
        stat.total_cost_usd += response.cost_usd
        stat.input_tokens += response.input_tokens
        stat.output_tokens += response.output_tokens

    @staticmethod
    def _split_provider_model(provider_model_key: str, fallback_model: str | None = None) -> tuple[str, str]:
        if "/" in provider_model_key:
            provider_name, model_name = provider_model_key.split("/", 1)
            return provider_name, model_name
        if fallback_model is None:
            raise ValueError("Model name is required")
        return provider_model_key, fallback_model

