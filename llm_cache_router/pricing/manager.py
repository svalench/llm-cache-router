from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_URL = (
    "https://raw.githubusercontent.com/svalench/llm-cache-router/main/"
    "llm_cache_router/pricing/pricing.json"
)
_DEFAULT_PRICE = {"input": 0.0, "output": 0.0}


def _bundled_pricing_path() -> Path:
    return Path(__file__).with_name("pricing.json")


def _normalize_payload(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for model_key, value in payload.items():
        if model_key.startswith("_"):
            continue
        if not isinstance(value, dict):
            continue
        normalized[model_key] = {
            "input": float(value.get("input", 0.0)),
            "output": float(value.get("output", 0.0)),
        }
    return normalized


class PricingManager:
    def __init__(
        self,
        remote_url: str = DEFAULT_REMOTE_URL,
        ttl_seconds: int = 3600,
        pricing_override: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._remote_url = remote_url
        self._ttl_seconds = ttl_seconds
        self._pricing_override = pricing_override or {}
        self._data = self._load_bundled()
        self._data.update(self._pricing_override)
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()

    async def ensure_fresh(self) -> None:
        now = time.monotonic()
        if now - self._loaded_at <= self._ttl_seconds:
            return

        async with self._lock:
            now = time.monotonic()
            if now - self._loaded_at <= self._ttl_seconds:
                return
            await self._fetch_remote()

    async def _fetch_remote(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self._remote_url)
                response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Remote pricing payload must be an object")
            data = _normalize_payload(payload)
            data.update(self._pricing_override)
            self._data = data
            self._loaded_at = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to refresh pricing data: %s", exc)

    def get(self, model_key: str) -> dict[str, float]:
        pricing = self._data.get(model_key, _DEFAULT_PRICE)
        return {"input": pricing["input"], "output": pricing["output"]}

    @property
    def all(self) -> dict[str, dict[str, float]]:
        return {
            key: {"input": value["input"], "output": value["output"]}
            for key, value in self._data.items()
        }

    async def sync_and_save(self) -> None:
        await self._fetch_remote()
        payload: dict[str, Any] = {
            "_meta": {
                "updated_at": date.today().isoformat(),
                "source": self._remote_url,
            }
        }
        payload.update(self._data)
        _bundled_pricing_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _load_bundled(self) -> dict[str, dict[str, float]]:
        try:
            payload = json.loads(_bundled_pricing_path().read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Bundled pricing payload must be an object")
            return _normalize_payload(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load bundled pricing data: %s", exc)
            return {}


_default_manager: PricingManager | None = None


def get_pricing_manager(**kwargs: Any) -> PricingManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = PricingManager(**kwargs)
    return _default_manager
