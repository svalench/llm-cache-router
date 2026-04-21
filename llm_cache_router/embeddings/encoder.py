from __future__ import annotations

from typing import Any, Protocol

import numpy as np

try:
    from sentence_transformers import (
        SentenceTransformer as _SentenceTransformerClass,  # type: ignore[import-untyped]
    )
except ImportError:  # pragma: no cover
    _SentenceTransformerClass = None

SentenceTransformerClass: Any | None = _SentenceTransformerClass


class EncoderProtocol(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class SentenceEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if SentenceTransformerClass is None:  # pragma: no cover
            raise RuntimeError("sentence-transformers is not installed")
        self._model: Any = SentenceTransformerClass(model_name)

    def encode(self, text: str) -> np.ndarray:
        embedding = self._model.encode([text], normalize_embeddings=True)[0]
        return np.array(embedding, dtype=np.float32)


class HashingEncoder:
    """
    Лёгкий deterministic encoder без внешних зависимостей для локальной разработки и тестов.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        if not text:
            return vec
        for token in text.lower().split():
            idx = abs(hash(token)) % self._dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
