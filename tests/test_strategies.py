from __future__ import annotations

import pytest

from llm_cache_router.strategies.cheapest import CheapestFirstStrategy
from llm_cache_router.strategies.fallback import AllProvidersFailedError, FallbackChainStrategy
from llm_cache_router.strategies.fastest import FastestFirstStrategy


def _known_pricing(monkeypatch: pytest.MonkeyPatch, pricing: dict[str, dict[str, float]]) -> None:
    from llm_cache_router.pricing import manager as pricing_manager_module

    class StubPricingManager:
        async def ensure_fresh(self) -> None:
            return None

        def get_or_none(self, model_key: str) -> dict[str, float] | None:
            return pricing.get(model_key)

        def get(self, model_key: str) -> dict[str, float]:
            value = pricing.get(model_key)
            if value is None:
                return {"input": 0.0, "output": 0.0}
            return value

    monkeypatch.setattr(pricing_manager_module, "get_pricing_manager", lambda: StubPricingManager())


@pytest.mark.asyncio
async def test_cheapest_selects_lowest_input_price(monkeypatch: pytest.MonkeyPatch) -> None:
    _known_pricing(
        monkeypatch,
        {
            "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "openai/gpt-4o": {"input": 2.5, "output": 10.0},
        },
    )
    strategy = CheapestFirstStrategy()

    selected = await strategy.select([("openai", "gpt-4o"), ("openai", "gpt-4o-mini")])

    assert selected == ("openai", "gpt-4o-mini")


@pytest.mark.asyncio
async def test_cheapest_tie_returns_first_option(monkeypatch: pytest.MonkeyPatch) -> None:
    _known_pricing(
        monkeypatch,
        {
            "openai/gpt-4o": {"input": 2.5, "output": 10.0},
            "azure/gpt-4o": {"input": 2.5, "output": 10.0},
        },
    )
    strategy = CheapestFirstStrategy()

    selected = await strategy.select([("openai", "gpt-4o"), ("azure", "gpt-4o")])

    # min() стабилен: при равной цене побеждает первый вариант
    assert selected == ("openai", "gpt-4o")


@pytest.mark.asyncio
async def test_cheapest_unknown_pricing_is_not_free(monkeypatch: pytest.MonkeyPatch) -> None:
    # Раньше модель без прайсинга получала цену 0.0 и всегда выигрывала.
    _known_pricing(monkeypatch, {"openai/gpt-4o-mini": {"input": 0.15, "output": 0.6}})
    strategy = CheapestFirstStrategy()

    selected = await strategy.select(
        [("openai_compatible", "self-hosted-model"), ("openai", "gpt-4o-mini")]
    )

    assert selected == ("openai", "gpt-4o-mini")


@pytest.mark.asyncio
async def test_cheapest_all_unknown_pricing_returns_first(monkeypatch: pytest.MonkeyPatch) -> None:
    _known_pricing(monkeypatch, {})
    strategy = CheapestFirstStrategy()

    selected = await strategy.select([("vllm", "model-a"), ("vllm", "model-b")])

    assert selected == ("vllm", "model-a")


@pytest.mark.asyncio
async def test_cheapest_empty_options_raises() -> None:
    strategy = CheapestFirstStrategy()

    with pytest.raises(ValueError, match="No providers available"):
        await strategy.select([])


def test_fastest_unknown_latency_defaults_to_slow() -> None:
    strategy = FastestFirstStrategy()
    strategy.observe("openai/gpt-4o", 100)

    # Неизвестный провайдер получает штраф 10 секунд и не выбирается
    assert strategy.select(["openai/gpt-4o", "unknown/model"]) == "openai/gpt-4o"


def test_fastest_ema_converges_to_faster_provider() -> None:
    strategy = FastestFirstStrategy()
    for _ in range(20):
        strategy.observe("slow/model", 1000)
        strategy.observe("fast/model", 100)

    assert strategy.select(["slow/model", "fast/model"]) == "fast/model"


def test_fastest_tie_returns_first() -> None:
    strategy = FastestFirstStrategy()
    strategy.observe("a/model", 100)
    strategy.observe("b/model", 100)

    assert strategy.select(["a/model", "b/model"]) == "a/model"


def test_fastest_empty_options_raises() -> None:
    strategy = FastestFirstStrategy()

    with pytest.raises(ValueError, match="No providers available"):
        strategy.select([])


@pytest.mark.asyncio
async def test_fallback_returns_first_success() -> None:
    strategy = FallbackChainStrategy(chain=["a/m1", "b/m2"])

    async def call(provider_model: str) -> str:
        return f"ok:{provider_model}"

    result = await strategy.execute(call)
    assert result == "ok:a/m1"


@pytest.mark.asyncio
async def test_fallback_skips_failed_providers() -> None:
    strategy = FallbackChainStrategy(chain=["a/m1", "b/m2"])
    calls: list[str] = []

    async def call(provider_model: str) -> str:
        calls.append(provider_model)
        if provider_model == "a/m1":
            raise RuntimeError("provider down")
        return f"ok:{provider_model}"

    result = await strategy.execute(call)
    assert result == "ok:b/m2"
    assert calls == ["a/m1", "b/m2"]


@pytest.mark.asyncio
async def test_fallback_exhausted_chain_raises_with_errors() -> None:
    strategy = FallbackChainStrategy(chain=["a/m1", "b/m2"])

    async def call(provider_model: str) -> str:
        raise RuntimeError(f"{provider_model} failed")

    with pytest.raises(AllProvidersFailedError, match="a/m1.*b/m2"):
        await strategy.execute(call)


@pytest.mark.asyncio
async def test_fallback_empty_chain_raises() -> None:
    strategy = FallbackChainStrategy(chain=[])

    async def call(provider_model: str) -> str:
        raise AssertionError("should not be called")

    with pytest.raises(AllProvidersFailedError, match="All providers failed"):
        await strategy.execute(call)


@pytest.mark.asyncio
async def test_fallback_timeout_falls_through() -> None:
    import asyncio

    strategy = FallbackChainStrategy(chain=["slow/m1", "fast/m2"], timeout=0.05)

    async def call(provider_model: str) -> str:
        if provider_model == "slow/m1":
            await asyncio.sleep(1.0)
            raise AssertionError("should be cancelled")
        return "ok:fast/m2"

    result = await strategy.execute(call)
    assert result == "ok:fast/m2"
