# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/milu/`. Major packages separate agent orchestration (`agent/`), LLM providers (`llm/providers/`), tools and MCP integrations (`tools/`), serving and channels (`serving/`, `channels/`), and supporting features such as scheduling, knowledge, and sandboxing. Tests mirror these areas under `tests/`; MCP-specific tests are in `tests/test_mcp/`. Runnable examples belong in `examples/`, configuration templates in `config/`, documentation in `docs/`, and screenshots or diagrams in `assets/`.

## Build, Test, and Development Commands

- `python -m venv .venv` creates a local environment; activate it, then run `pip install -e ".[dev]"` for an editable install with test and lint tools.
- `ruff check src/ tests/` runs the same static checks used by CI.
- `python -m pytest tests/ --ignore=tests/test_real_api.py --ignore=tests/test_real_new_providers.py -q` runs the full offline unit suite.
- `python -m pytest tests/test_agent.py -v` runs one focused test module.
- `milu serve` starts the local web service after installation. Alternatively, `docker compose up --build` builds and runs it on port 8000.

## Coding Style & Naming Conventions

Target Python 3.10+, use four-space indentation, modern type syntax such as `str | None`, and `@dataclass` for data models (`frozen=True` when immutable). Ruff enforces Pyflakes and selected pycodestyle rules; keep lines near the configured 100-character limit. Follow `snake_case` for modules/functions, `PascalCase` for classes, and `{PROVIDER}_API_KEY` for provider credentials. Existing code commonly uses Chinese comments/docstrings; keep terminology consistent with the file being edited.

## Testing Guidelines

Use pytest with `asyncio_mode = "auto"`; async tests need no marker. Name files `test_<feature>.py` and tests `test_<behavior>`. Mock provider calls with `unittest.mock.AsyncMock` and fixtures from `tests/conftest.py`. Real-API tests require keys in `.env` and should not be part of routine CI runs. Add regression tests for every bug fix, especially around concurrency and per-user isolation.

## Commit & Pull Request Guidelines

Use a short English, imperative subject, matching history (for example, `Fix sandbox timeout leaking orphan process tree`); scoped Conventional Commit forms such as `feat(agent): ...` and `chore: ...` are also accepted. Keep commits focused. PRs must explain what and why, link related issues, pass Ruff and unit tests, update both READMEs when behavior changes, and contain no secrets. Include screenshots for dashboard or web UI changes.

## Security & Configuration

Copy `.env.example` to `.env` and never commit real API keys. Keep machine-specific MCP settings in ignored `config/mcp_servers.json`; commit only the example template.
