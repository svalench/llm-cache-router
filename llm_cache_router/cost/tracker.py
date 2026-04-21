from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from llm_cache_router.models import TokenUsage
from llm_cache_router.strategies.cheapest import PRICING


class BudgetExceededError(RuntimeError):
    pass


class CostTracker:
    def __init__(self, budget: dict | None = None) -> None:
        self.budget = budget or {}
        self._daily_spend = 0.0
        self._monthly_spend = 0.0
        self._total_spend = 0.0
        self._reset_at = datetime.now(UTC).date()
        self._month_reset_at = datetime.now(UTC).strftime("%Y-%m")
        self._lock = asyncio.Lock()

    async def record(self, provider: str, model: str, usage: TokenUsage) -> float:
        return await self._record_async(provider, model, usage)

    def __getattribute__(self, name: str):  # noqa: ANN204
        if name == "record":
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return object.__getattribute__(self, "_record_sync")
        return object.__getattribute__(self, name)

    async def _record_async(self, provider: str, model: str, usage: TokenUsage) -> float:
        async with self._lock:
            self._check_reset()
            cost = self._calculate(provider, model, usage)
            self._daily_spend += cost
            self._monthly_spend += cost
            self._total_spend += cost

            daily_limit = self.budget.get("daily_usd")
            if daily_limit is not None and self._daily_spend > float(daily_limit):
                raise BudgetExceededError(
                    f"Daily budget ${daily_limit} exceeded. Spent: ${self._daily_spend:.6f}"
                )
            monthly_limit = self.budget.get("monthly_usd")
            if monthly_limit is not None and self._monthly_spend > float(monthly_limit):
                raise BudgetExceededError(
                    f"Monthly budget ${monthly_limit} exceeded. Spent: ${self._monthly_spend:.6f}"
                )
            return cost

    def _record_sync(self, provider: str, model: str, usage: TokenUsage) -> float:
        return asyncio.run(self._record_async(provider, model, usage))

    def stats(self) -> dict[str, float | None]:
        daily_limit = self.budget.get("daily_usd")
        monthly_limit = self.budget.get("monthly_usd")
        remaining = None
        monthly_remaining = None
        if daily_limit is not None:
            remaining = max(0.0, float(daily_limit) - self._daily_spend)
        if monthly_limit is not None:
            monthly_remaining = max(0.0, float(monthly_limit) - self._monthly_spend)
        return {
            "daily_spend_usd": round(self._daily_spend, 6),
            "monthly_spend_usd": round(self._monthly_spend, 6),
            "total_spend_usd": round(self._total_spend, 6),
            "budget_remaining_usd": round(remaining, 6) if remaining is not None else None,
            "monthly_budget_remaining_usd": (
                round(monthly_remaining, 6) if monthly_remaining is not None else None
            ),
        }

    def _check_reset(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._reset_at:
            self._daily_spend = 0.0
            self._reset_at = today
        current_month = datetime.now(UTC).strftime("%Y-%m")
        if current_month != self._month_reset_at:
            self._monthly_spend = 0.0
            self._month_reset_at = current_month

    @staticmethod
    def _calculate(provider: str, model: str, usage: TokenUsage) -> float:
        key = f"{provider}/{model}"
        pricing = PRICING.get(key, {"input": 0.0, "output": 0.0})
        in_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        out_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        return in_cost + out_cost
