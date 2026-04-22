# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
