from __future__ import annotations

import numpy as np

from llm_cache_router.embeddings.encoder import HashingEncoder


def test_hashing_encoder_deterministic() -> None:
    encoder = HashingEncoder(dimension=64)

    first = encoder.encode("семантическй кэш для llm")
    second = encoder.encode("семантическй кэш для llm")

    np.testing.assert_array_equal(first, second)


def test_hashing_encoder_dimension() -> None:
    encoder = HashingEncoder(dimension=128)

    vec = encoder.encode("любой текст")

    assert vec.shape == (128,)
    assert vec.dtype == np.float32


def test_hashing_encoder_normalizes() -> None:
    encoder = HashingEncoder(dimension=64)

    vec = encoder.encode("нормализация вектора эмбеддинга")

    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_hashing_encoder_empty_text_returns_zero_vector() -> None:
    encoder = HashingEncoder(dimension=64)

    vec = encoder.encode("")

    assert vec.shape == (64,)
    assert float(np.linalg.norm(vec)) == 0.0


def test_hashing_encoder_case_insensitive() -> None:
    encoder = HashingEncoder(dimension=64)

    lower = encoder.encode("semantic cache for llm routing")
    upper = encoder.encode("Semantic Cache For LLM Routing")

    np.testing.assert_array_equal(lower, upper)


def test_hashing_encoder_different_texts_are_different() -> None:
    encoder = HashingEncoder(dimension=256)

    first = encoder.encode("как сбросить пароль")
    second = encoder.encode("как настроить кэширование")

    cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
    assert cosine < 0.999


def test_hashing_encoder_out_of_vocabulary_text() -> None:
    # OOV-токены — это просто неизвестные хэши: вектор всё равно
    # детерминирован, имеет корректную размерность и норму 1.
    encoder = HashingEncoder(dimension=384)

    vec = encoder.encode("zzqxj wvknrp 12345 !@#$% привет世界")

    assert vec.shape == (384,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6
    np.testing.assert_array_equal(vec, encoder.encode("zzqxj wvknrp 12345 !@#$% привет世界"))
