# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-09-11

Maintenance release adding an offline onboarding path and stronger distribution validation.

- Add a packaged offline demo using a stub provider and the real in-memory cache, without API keys or paid model calls.
- Restore the missing environment-variable template and improve CLI help.
- Correct documentation about dependencies, provider registration, budgets, and demo limitations.
- Add network-isolated regression tests and an installed-wheel smoke check.

Details: [v0.3.1 release notes](docs/releases/v0.3.1.md).

## [0.3.0] - 2026-09-04

### Added
- `openai_compatible` provider: any OpenAI Chat Completions endpoint via `base_url` (OpenRouter, vLLM, llama.cpp server, LiteLLM proxy, self-hosted inference). `api_key` optional for keyless local servers.
- Cache invalidation API: `LLMRouter.invalidate_cache(model=...)` and `invalidate(model=...)` on all backends (memory / Redis / Qdrant).
- `CacheConfig.key_version` — cache key versioning: bump the version to make old entries invisible after a prompt deploy, without flushing.
- `CacheConfig.exact_match` — exact-match-only cache mode (semantic search disabled) for correctness-sensitive workloads.
- CONTRIBUTING.md, issue templates (bug report / feature request) and a PR template.

### Changed
- `CHEAPEST_FIRST` now treats models with unknown pricing as infinitely expensive instead of free, so self-hosted models no longer win routing by default.
- README: new «Cache Invalidation, Versioning & Exact Match» section (incl. threshold false-positive guidance), OpenAI-compatible provider docs, roadmap reordered (OpenTelemetry before Django helpers).
- Removed committed `llm_cache_router.egg-info/` build artifact and `.cursor/` scratchpad from the repository; `uv.lock` excluded from sdist via MANIFEST.in.

### Fixed
- OpenAI provider no longer sends `Authorization: Bearer None` when `api_key` is not set.

## [0.2.4] - 2026-05-24

### Added
- `Message` type alias for multimodal chat payloads.
- `CacheEntry.model` and `model` parameter on cache `get`/`set` for strict per-model isolation.
- Media-aware cache key normalization in `CacheBackend._messages_to_text` (image/audio/video hashes, no full base64 in `query`).
- Qdrant search filter by `model` when provided.

### Changed
- Router passes requested `model` into all cache lookups and stores (including stream warmup).
- Provider APIs use `list[Message]`; Gemini text extraction reads only `text` blocks from list content.

### Fixed
- Multimodal messages no longer collapse to `str(list)` in cache keys.
- Same prompt with different models no longer returns a cross-model cache hit.

## [0.2.3] - 2026-04-22

### Fixed
- Corrected author/maintainer name in `pyproject.toml` to `Alexander Valenchits`.

## [0.2.2] - 2026-04-21

### Added
- Full PyPI-ready metadata in `pyproject.toml`: author, license, keywords, classifiers, project URLs.
- `LICENSE` file (MIT).
- `CHANGELOG.md`.
- `py.typed` marker for PEP 561 type-checker support.
- GitHub Actions workflow to publish wheels and sdist to PyPI on tag push (`publish.yml`).
- Greatly expanded README with badges, TOC, feature list, provider matrix and contributing section.

### Changed
- `pyproject.toml` — explicit `setuptools.packages.find` include/exclude.
- Documentation restructured to lead with English (PyPI audience) with Russian summary at the bottom.

### Fixed
- Packaging excludes `tests/` and `examples/` from the shipped wheel.

## [0.2.1] - 2026-04

### Fixed
- Removed config duplication.
- Fixed fallback exception handling in streaming.
- Made `CostTracker` lock async.

## [0.2.0] - 2026-04

### Added
- First tagged release.
- Providers: OpenAI, Anthropic, Gemini, Ollama, MiniMax, Qwen.
- Cache backends: in-memory (FAISS), Redis, Qdrant.
- Routing strategies: cheapest / fastest / fallback.
- Cost tracker with budget limits.
- FastAPI middleware + Prometheus metrics.
- Cache warmup API.
