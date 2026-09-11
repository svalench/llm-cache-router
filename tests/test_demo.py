from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock

import pytest

from llm_cache_router import cli, demo
from llm_cache_router.embeddings import encoder
from llm_cache_router.pricing.manager import get_pricing_manager


@pytest.mark.asyncio
async def test_demo_is_offline_and_reuses_cached_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_model(*args: object, **kwargs: object) -> None:
        pytest.fail("The demo must not instantiate an embedding model")

    async def no_pricing() -> None:
        pytest.fail("The demo must not refresh remote pricing")

    monkeypatch.setattr(encoder, "sentence_transformer_class", no_model)
    monkeypatch.setattr(get_pricing_manager(), "ensure_fresh", no_pricing)
    # Even when an API key is present, the demo must never select a real provider.
    monkeypatch.setenv("OPENAI_API_KEY", "unused-demo-test-key")
    complete = AsyncMock(
        side_effect=lambda **kwargs: demo.LLMResponse(
            content="stub response", provider_used="offline_demo", model_used=kwargs["model"]
        )
    )
    close = AsyncMock()
    monkeypatch.setattr(demo.DemoProvider, "complete", complete)
    monkeypatch.setattr(demo.DemoProvider, "close", close)

    for _ in range(2):
        result = await demo.run_demo()
        assert result["first_cache_hit"] is False
        assert result["second_cache_hit"] is True
        assert result["provider_calls"] == 1
        assert result["total_requests"] == 2
        assert result["cache_hits"] == 1
        assert result["total_cost_usd"] == 0.0
        assert result["response"] == "stub response"

    assert complete.await_count == 2
    assert close.await_count == 2


def test_cli_demo_runs_without_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["llm-cache-router", "demo"])

    cli.main()

    result = json.loads(capsys.readouterr().out)
    assert result["first_cache_hit"] is False
    assert result["second_cache_hit"] is True
    assert result["provider_calls"] == 1
    assert "fixed demo response" in result["response"]


def test_cli_help_lists_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["llm-cache-router", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "demo" in output
    assert "pricing-sync" in output


@pytest.mark.parametrize("args", [[], ["unknown"], ["demo", "--api-key", "not-accepted"]])
def test_cli_rejects_invalid_arguments(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["llm-cache-router", *args])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_cli_pricing_sync_is_preserved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sync = AsyncMock()
    monkeypatch.setattr(get_pricing_manager(), "sync_and_save", sync)
    monkeypatch.setattr(sys, "argv", ["llm-cache-router", "pricing-sync"])

    cli.main()

    sync.assert_awaited_once()
    assert "synced successfully" in capsys.readouterr().out
