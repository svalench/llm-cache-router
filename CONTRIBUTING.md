# Contributing to llm-cache-router

Thanks for your interest in contributing. This document covers the basics; if anything is unclear, open an issue and ask.

## Ways to contribute

- **Bug reports** — open an issue using the bug report template. Include a minimal reproduction.
- **Feature requests** — open an issue using the feature request template first, before writing code.
- **Good first issues** — look for the [`good first issue`](https://github.com/svalench/llm-cache-router/labels/good%20first%20issue) label.
- **Questions / ideas** — [GitHub Discussions](https://github.com/svalench/llm-cache-router/discussions) is the right place.

## Development setup

```bash
git clone https://github.com/svalench/llm-cache-router.git
cd llm-cache-router

# using uv (recommended)
uv sync --all-extras
uv run pytest

# or plain pip
pip install -e ".[all,dev]"
pytest
```

## Making changes

1. Fork / branch from `main`.
2. Keep PRs focused — one feature or fix per PR.
3. Add tests for any new behaviour. Bug fixes should include a regression test.
4. Keep the public API typed: Pydantic models for data, type hints everywhere.
5. Match the existing code style — comments in Russian are fine, identifiers in English.

## Before you push

CI runs these on every PR — run them locally first:

```bash
ruff check llm_cache_router/ tests/
ruff format llm_cache_router/ tests/
mypy llm_cache_router/ --ignore-missing-imports
pytest
```

## Adding a provider

1. Subclass `LLMProvider` in `llm_cache_router/providers/<name>.py`.
2. Implement `complete()` (streaming is optional — the base class falls back to a single-chunk stream).
3. Register it: `register_provider("<name>", YourProvider)`.
4. Add the import to `llm_cache_router/providers/__init__.py`.
5. Add tests with a mocked HTTP transport (see `tests/test_openai_compatible.py` for the pattern).
6. Update the provider table in `README.md`.

## Adding a cache backend

1. Subclass `CacheBackend` — implement `get`, `set`, `clear`, `invalidate`.
2. Wire it into `LLMRouter._build_cache` and add it to the optional dependencies in `pyproject.toml`.
3. Cover it with tests using a fake client (see `tests/test_redis_cache.py` for the pattern).

## Releasing (maintainers)

Releases follow [SemVer](https://semver.org/). Update `CHANGELOG.md`, bump the version in `pyproject.toml`, tag, and let CI publish to PyPI.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
