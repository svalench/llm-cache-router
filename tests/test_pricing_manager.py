from __future__ import annotations

import time
from unittest.mock import AsyncMock

import httpx
import pytest

from llm_cache_router.pricing.manager import PricingManager


def test_pricing_manager_loads_bundled_data() -> None:
    manager = PricingManager()
    assert manager.get("openai/gpt-4o")["input"] == 2.50


def test_pricing_override_takes_priority() -> None:
    manager = PricingManager(
        pricing_override={"openai/gpt-4o": {"input": 1.00, "output": 5.00}}
    )
    assert manager.get("openai/gpt-4o")["input"] == 1.00


@pytest.mark.asyncio
async def test_ensure_fresh_skips_fetch_when_ttl_not_expired() -> None:
    manager = PricingManager(ttl_seconds=3600)
    manager._fetch_remote = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
    manager._loaded_at = time.monotonic()  # noqa: SLF001

    await manager.ensure_fresh()

    manager._fetch_remote.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001


@pytest.mark.asyncio
async def test_ensure_fresh_fetches_when_ttl_expired() -> None:
    manager = PricingManager(ttl_seconds=3600)
    manager._fetch_remote = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
    manager._loaded_at = 0.0  # noqa: SLF001

    await manager.ensure_fresh()

    manager._fetch_remote.assert_awaited_once()  # type: ignore[attr-defined]  # noqa: SLF001


@pytest.mark.asyncio
async def test_fetch_remote_network_error_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = PricingManager()
    before = manager.all

    async def _raise_connect_error(*args: object, **kwargs: object) -> object:  # noqa: ARG001
        raise httpx.ConnectError("network error")

    monkeypatch.setattr(httpx.AsyncClient, "get", _raise_connect_error)

    await manager._fetch_remote()  # noqa: SLF001

    assert manager.all == before


def test_get_unknown_model_returns_zero_cost() -> None:
    manager = PricingManager()
    assert manager.get("unknown/model") == {"input": 0.0, "output": 0.0}
