from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from llm_cache_router.models import CacheEntry, LLMResponse, Message


class CacheBackend(ABC):
    @abstractmethod
    async def get(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
    ) -> tuple[CacheEntry | None, float | None]:
        raise NotImplementedError

    @abstractmethod
    async def set(
        self,
        messages: list[Message],
        response: LLMResponse,
        *,
        model: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        raise NotImplementedError

    async def invalidate(self, *, model: str | None = None) -> int:
        """Инвалидация записей кэша. Возвращает число удалённых записей.

        Если ``model`` передан — удаляются только записи этой модели,
        иначе — все записи. Бэкенды, не поддерживающие выборочное удаление,
        могут выбросить NotImplementedError.
        """
        raise NotImplementedError("This cache backend does not support invalidate()")

    async def close(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {}

    @staticmethod
    def _model_matches(entry_model: str | None, requested_model: str | None) -> bool:
        if requested_model is None:
            return True
        return entry_model == requested_model

    @staticmethod
    def _short_hash(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    @classmethod
    def _hash_media_value(cls, value: str, prefix: str) -> str:
        return f"{prefix}:{cls._short_hash(value)}"

    @classmethod
    def _media_fingerprint_from_block(cls, block: dict[str, Any]) -> str | None:
        btype = block.get("type", "")
        if btype == "text":
            return None
        if btype == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if not url:
                return None
            if url.startswith("data:"):
                data_part = url.split(",", 1)[-1] if "," in url else url
                return cls._hash_media_value(data_part, "img")
            return cls._hash_media_value(url, "img_url")
        if btype == "image":
            source = block.get("source", {})
            data = source.get("data", "")
            if data:
                return cls._hash_media_value(data, "img")
            return None
        if btype in {"input_audio", "audio"}:
            audio = block.get("input_audio") or block.get("audio") or {}
            data = audio.get("data", "")
            if data:
                return cls._hash_media_value(data, "audio")
            return None
        if btype in {"video", "input_video"}:
            video = block.get("video") or block.get("input_video") or {}
            url = video.get("url", "")
            data = video.get("data", "")
            if data:
                return cls._hash_media_value(data, "video")
            if url:
                return cls._hash_media_value(url, "video_url")
            return None
        # Неизвестный блок — стабильный хэш JSON без полного base64 в query
        return cls._hash_media_value(json.dumps(block, sort_keys=True, default=str), "block")

    @classmethod
    def _content_to_text(cls, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        text_parts: list[str] = []
        media_hashes: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block.get("text", ""))
                continue
            fingerprint = cls._media_fingerprint_from_block(block)
            if fingerprint:
                media_hashes.append(fingerprint)

        combined = " ".join(text_parts).strip()
        if media_hashes:
            suffix = " [" + ",".join(media_hashes) + "]"
            combined = (combined + suffix) if combined else suffix.strip()
        return combined

    @classmethod
    def _messages_to_text(cls, messages: list[Message]) -> str:
        chunks: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            normalized = cls._content_to_text(content)
            chunks.append(f"{role}:{normalized}")
        return "\n".join(chunks).strip()
