from __future__ import annotations

import asyncio
import sys


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in ("pricing-sync",):
        print("Usage: llm-cache-router pricing-sync")
        sys.exit(1)

    if args[0] == "pricing-sync":
        from llm_cache_router.pricing.manager import get_pricing_manager

        asyncio.run(get_pricing_manager().sync_and_save())
        print("pricing.json synced successfully")
