from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-cache-router")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run an offline stub-provider cache demo (no API key)")
    commands.add_parser("pricing-sync", help="Fetch remote pricing and update bundled pricing.json")
    args = parser.parse_args()

    if args.command == "demo":
        from llm_cache_router.demo import main as demo_main

        demo_main()
    elif args.command == "pricing-sync":
        from llm_cache_router.pricing.manager import get_pricing_manager

        asyncio.run(get_pricing_manager().sync_and_save())
        print("pricing.json synced successfully")


if __name__ == "__main__":
    main()
