from __future__ import annotations

import json

import httpx
import pytest

from llm_cache_router.providers.base import ProviderConfig, ProviderError
from llm_cache_router.providers.openai_compatible import OpenAICompatibleProvider


def _provider(
    base_url: str = "http://localhost:8000/v1",
    api_key: str | None = "test-key",
    transport: httpx.MockTransport | None = None,
) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        ProviderConfig(name="openai_compatible", base_url=base_url, api_key=api_key)
    )
    if transport is not None:
        provider._client = httpx.AsyncClient(transport=transport)
    return provider


def test_missing_base_url_raises() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleProvider(ProviderConfig(name="openai_compatible", api_key="k"))


def test_api_key_is_optional_for_local_servers() -> None:
    provider = _provider(api_key=None)
    assert provider is not None


def test_registered_in_registry() -> None:
    import llm_cache_router.providers  # noqa: F401
    from llm_cache_router.providers.registry import get_provider_class

    assert get_provider_class("openai_compatible") is OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_complete_uses_custom_base_url() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "local answer"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )

    provider = _provider(transport=httpx.MockTransport(handler))
    response = await provider.complete(
        messages=[{"role": "user", "content": "привет"}],
        model="qwen2.5-32b-instruct",
        temperature=0.1,
        max_tokens=100,
    )

    assert response.content == "local answer"
    assert response.provider_used == "openai_compatible"
    assert response.model_used == "qwen2.5-32b-instruct"
    assert response.input_tokens == 11
    assert response.output_tokens == 7

    assert len(requests) == 1
    assert str(requests[0].url) == "http://localhost:8000/v1/chat/completions"
    body = json.loads(requests[0].content.decode())
    assert body["model"] == "qwen2.5-32b-instruct"
    assert body["max_tokens"] == 100
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    await provider.close()


@pytest.mark.asyncio
async def test_complete_without_api_key_sends_no_auth_header() -> None:
    headers_seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers_seen.append(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(api_key=None, transport=httpx.MockTransport(handler))
    response = await provider.complete(
        messages=[{"role": "user", "content": "hi"}], model="llama-3.1-8b"
    )

    assert response.content == "ok"
    assert "Authorization" not in headers_seen[0]
    await provider.close()


@pytest.mark.asyncio
async def test_complete_raises_provider_error_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(500, text="boom")

    provider = _provider(transport=httpx.MockTransport(handler))

    with pytest.raises(ProviderError, match="500"):
        await provider.complete(messages=[{"role": "user", "content": "hi"}], model="m")
    await provider.close()


@pytest.mark.asyncio
async def test_stream_parses_sse_from_custom_endpoint() -> None:
    sse = "\n".join(
        [
            'data: {"choices": [{"delta": {"content": "Hel"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"content": "lo"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}',
            "data: [DONE]",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    provider = _provider(transport=httpx.MockTransport(handler))
    chunks = [
        chunk
        async for chunk in provider.stream(
            messages=[{"role": "user", "content": "hi"}], model="local-model"
        )
    ]

    assert "".join(chunk.delta for chunk in chunks) == "Hello"
    assert chunks[-1].is_final
    assert chunks[-1].input_tokens == 3
    assert chunks[-1].output_tokens == 2
    assert all(chunk.provider_used == "openai_compatible" for chunk in chunks)
    await provider.close()


@pytest.mark.asyncio
async def test_base_url_trailing_slash_is_normalized() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(
        base_url="http://localhost:8000/v1/", transport=httpx.MockTransport(handler)
    )
    await provider.complete(messages=[{"role": "user", "content": "hi"}], model="m")

    assert urls == ["http://localhost:8000/v1/chat/completions"]
    await provider.close()
