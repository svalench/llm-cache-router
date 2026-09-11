from __future__ import annotations

import socket
from unittest.mock import AsyncMock

import pytest

from llm_cache_router.pricing.manager import get_pricing_manager


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests must use fake clients/MockTransport, never live services."""

    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail("Network access is disabled in unit tests; use a fake client or MockTransport")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture(autouse=True)
def bundled_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Router tests share this singleton. Dedicated PricingManager tests create
    # their own instances, so refresh/TTL/error behavior remains under test.
    monkeypatch.setattr(get_pricing_manager(), "ensure_fresh", AsyncMock())
