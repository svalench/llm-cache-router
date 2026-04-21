from __future__ import annotations

import asyncio

from llm_cache_router import LLMRouter
from llm_cache_router.warmup import load_warmup_entries


async def main() -> None:
    async with LLMRouter(
        providers={"openai": {"api_key": "sk-...", "models": ["gpt-4o-mini"]}},
    ) as router:
        entries = load_warmup_entries("warmup_queries.json")
        stats = await router.warmup(entries, concurrency=3)
        print(
            f"Warmed: {stats['warmed']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
