from __future__ import annotations

import pytest

from llm_cache_router.cost.tracker import BudgetExceededError, CostTracker
from llm_cache_router.models import TokenUsage


def test_cost_tracker_records_and_exposes_monthly_stats() -> None:
    tracker = CostTracker(budget={"daily_usd": 1.0, "monthly_usd": 10.0})
    cost = tracker.record(
        provider="openai",
        model="gpt-4o-mini",
        usage=TokenUsage(input_tokens=100_000, output_tokens=20_000),
    )
    stats = tracker.stats()

    assert cost > 0
    assert stats["daily_spend_usd"] is not None
    assert stats["monthly_spend_usd"] is not None
    assert stats["monthly_budget_remaining_usd"] is not None


def test_cost_tracker_raises_on_budget_exceeded() -> None:
    tracker = CostTracker(budget={"daily_usd": 0.001})

    with pytest.raises(BudgetExceededError):
        tracker.record(
            provider="openai",
            model="gpt-4o",
            usage=TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000),
        )

