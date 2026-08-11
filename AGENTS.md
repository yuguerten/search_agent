# Repository Guidelines

## Project Structure & Module Organization

This repository is currently a Python project scaffold containing only `.gitignore`. As the agentic research workflow is added, keep responsibilities separated:

- `src/` — application and agent implementation, organized by domain (`agents/`, `tools/`, `workflows/`, and `storage/`).
- `tests/` — automated tests mirroring the `src/` layout.
- `docs/` — architecture notes, prompts, and operating decisions.
- `.env.example` — documented configuration names without secrets.

Prefer small modules with explicit interfaces between dispatcher, researcher, critic, and persistence components.

## Build, Test, and Development Commands

Use a project virtual environment and run commands from the repository root. Once the Python packaging setup is present, the expected workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
```

Run the local application with the project’s documented entry point (for example, `python -m src.main`). Use a local PostgreSQL instance for integration tests and keep external API credentials in environment variables.

## Coding Style & Naming Conventions

Follow PEP 8, four-space indentation, and type hints for public functions and agent/tool boundaries. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Ruff is the formatter and linter; format code before committing. Keep prompts and agent configuration close to the agent that owns them, and make tool inputs/outputs structured and serializable.

## Testing Guidelines

Use pytest. Name files `test_<module>.py` and tests `test_<behavior>`. Unit-test routing, filtering, ranking, critic decisions, and loop exit conditions without network access. Mark arXiv, LiteLLM, and PostgreSQL tests as integration tests and isolate them behind fixtures or mocks.

## Security & Configuration Tips

Never commit API keys, database credentials, papers, or generated vector data. Update `.env.example` when adding configuration. Validate tool responses and enforce citation/year filters before persisting or presenting results.

## Commit & Pull Request Guidelines

Use concise imperative commits, such as `Add critic ranking workflow`. Pull requests should explain the behavior changed, include tests and configuration changes, document any prompt or schema changes, and note external services required for verification.
