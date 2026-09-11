"""Run with the wheel environment's `python -I tests/smoke_distribution.py`.

No pytest, source-tree imports, optional services, or ML stack are needed.
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import sys
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path


def no_network(*args: object, **kwargs: object) -> None:
    raise AssertionError("The installed-package demo must not access the network")


socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.getaddrinfo = no_network

import llm_cache_router  # noqa: E402
from llm_cache_router.pricing.manager import get_pricing_manager  # noqa: E402

assert Path(llm_cache_router.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), (
    "Smoke test must import the installed wheel, not the source checkout"
)
package = files("llm_cache_router")
assert package.joinpath("py.typed").is_file()
assert package.joinpath("demo.py").is_file()
pricing = json.loads(package.joinpath("pricing/pricing.json").read_text(encoding="utf-8"))
assert "openai/gpt-4o-mini" in pricing
assert get_pricing_manager().get("openai/gpt-4o-mini") == pricing["openai/gpt-4o-mini"]


# Fail even if pricing refresh would swallow a blocked connection error.
async def no_pricing() -> None:
    raise AssertionError("The installed-package demo must not refresh pricing")


get_pricing_manager().ensure_fresh = no_pricing
entrypoint = next(
    entry
    for entry in distribution("llm-cache-router").entry_points
    if entry.group == "console_scripts" and entry.name == "llm-cache-router"
)
sys.argv = ["llm-cache-router", "demo"]
output = io.StringIO()
with contextlib.redirect_stdout(output):
    entrypoint.load()()
result = json.loads(output.getvalue())
assert result["first_cache_hit"] is False
assert result["second_cache_hit"] is True
assert result["provider_calls"] == 1
assert result["total_requests"] == 2
assert result["cache_hits"] == 1
assert result["total_cost_usd"] == 0.0
assert "fixed demo response" in result["response"]
print(output.getvalue(), end="")
print("Installed-wheel CLI, offline demo, pricing.json, and py.typed smoke checks passed.")
