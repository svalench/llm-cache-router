from __future__ import annotations

import json
from pathlib import Path

from llm_cache_router.models import WarmupEntry


def load_warmup_entries(path: str | Path) -> list[WarmupEntry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [WarmupEntry(**item) for item in data]
