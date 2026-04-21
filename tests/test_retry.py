from __future__ import annotations

import pytest

from llm_cache_router.retry import RetryConfig, with_retry


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt() -> None:
    call_count = 0

    async def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("temporary error")
        return "ok"

    result = await with_retry(flaky, RetryConfig(attempts=3, base_delay_sec=0.01, jitter=False))
    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_raises_after_exhaustion() -> None:
    async def always_fail() -> str:
        raise Exception("permanent error")

    with pytest.raises(Exception, match="permanent error"):
        await with_retry(always_fail, RetryConfig(attempts=3, base_delay_sec=0.01, jitter=False))


@pytest.mark.asyncio
async def test_no_retry_on_4xx() -> None:
    class Http400Error(Exception):
        status_code = 400

    async def bad_request() -> str:
        raise Http400Error("bad request")

    with pytest.raises(Http400Error):
        await with_retry(bad_request, RetryConfig(attempts=3, jitter=False))
