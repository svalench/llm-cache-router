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

### Lightweight offline unit tests

The normal install includes sentence-transformers/PyTorch. To work on routing,
hash-based caching, or packaging without downloading that ML stack, use a fresh
environment and explicitly install only the dependencies needed by the mocked
unit suite:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "pydantic>=2.0" "httpx>=0.27" "numpy>=1.26" \
  "fastapi>=0.111" "pytest>=8.0" "pytest-asyncio>=0.23"
python -m pip install --no-deps -e .
python -m pytest
llm-cache-router demo
```

This deliberately omits declared ML dependencies: it is a **test-only setup**,
not a supported replacement for the normal installation or a semantic-embedding
test. `pip check` will report the omitted dependencies. Redis/Qdrant tests use
fake clients; the memory cache uses its NumPy fallback when FAISS is absent.
Optionally install `faiss-cpu` to exercise its index path without PyTorch.

Tests block outbound connections and use bundled pricing, hash embeddings, and
fake providers. Never add tests that require real API keys, model downloads, or
paid calls. Dedicated pricing tests mock their own HTTP calls.

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

CI also builds an sdist and a wheel from it, checks their metadata/README, and
smoke-tests the installed wheel in isolation from the source checkout:

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

The distribution smoke job intentionally installs only Pydantic, HTTPX, and NumPy
alongside the wheel with `--no-deps`; it checks the hash/NumPy fallback, CLI, and
bundled resources, not the complete dependency stack. The regular test job
retains the full dependency install.

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
