# Contributing to milu

Thanks for your interest in milu! Issues and PRs are welcome — in **English or 中文**.

## Development setup

```bash
git clone https://github.com/stephonGAO/milu.git
cd milu
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

> On Windows the venv interpreter is `.venv/Scripts/python` (not `.venv/bin/python`).

## Running tests

```bash
# full unit suite (skips tests that hit real APIs)
python -m pytest tests/ --ignore=tests/test_real_api.py --ignore=tests/test_real_new_providers.py -q

# a single file / method
python -m pytest tests/test_agent.py -v
```

`asyncio_mode = "auto"` is configured, so async tests need no `@pytest.mark.asyncio`. Mock LLM
responses with `unittest.mock.AsyncMock` — see the `mock_openai_client` fixture in `tests/conftest.py`.
Real-API integration tests (`tests/test_real_api.py`, `tests/test_real_new_providers.py`) require
provider keys in `.env`.

## Code conventions

- **Chinese docstrings/comments and Chinese commit messages** (matching the existing codebase).
- Python **3.10+** syntax: `str | None` unions, `match/case`.
- Data models use `@dataclass` (immutable ones add `frozen=True`).
- Env var naming: `{PROVIDER}_API_KEY` (e.g. `QWEN_API_KEY`, `ANTHROPIC_API_KEY`).
  **Never commit real keys** — `.env` is gitignored; copy `.env.example` and fill it in locally.
- No enforced linter/formatter (ruff / black / mypy are not configured).

## Adding an LLM provider

Each provider is a single file under `src/milu/llm/providers/`. All providers speak through
`openai.AsyncOpenAI` (different `base_url` + `extra_body`). Self-register at the end of the file:

```python
ModelRegistry.register("yourprovider", YourProviderClass)
```

Add a matching `tests/test_yourprovider.py`. Because unit tests fully mock the client, please also
**smoke-test a real basic conversation** before submitting — mock-only tests have previously missed
real request/response contract mismatches.

## Pull requests

1. Branch off `main`.
2. Keep the change focused; update `README.md` / `README.zh-CN.md` when behavior changes.
3. Make sure the unit suite passes.
4. Open the PR with a clear description and link any related issue.

Questions? Open an issue: <https://github.com/stephonGAO/milu/issues>
