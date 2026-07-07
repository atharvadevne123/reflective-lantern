# Contributing to Reflective Lantern

Thank you for your interest! This is a personal automation project, but
external contributions are welcome for bug fixes, new utility scripts, and
documentation improvements.

## Getting Started

```bash
git clone https://github.com/atharvadevne123/reflective-lantern.git
cd reflective-lantern
bash scripts/setup.sh
```

This installs all dev dependencies and pre-commit hooks.

## Development Workflow

```bash
make test       # run pytest
make lint       # ruff check
make format     # ruff format
make type-check # mypy
```

All four commands must pass before opening a pull request.

## Pull Request Guidelines

- **One change per PR** — keep diffs small and focused
- **Tests required** — new scripts must have corresponding `tests/test_*.py`
- **No secrets** — never commit API keys or passwords; use `.env.example`
- **Update `.env.example`** if you add new environment variables
- **Update `CHANGELOG.md`** with a brief description of your change

## Project Structure

```
reflective-lantern/
├── .claude/settings.json   ← CCR tool permissions
├── config/                  ← Python config package
├── scripts/                 ← Standalone utility scripts
├── tests/                   ← pytest test suite
├── docs/                    ← Architecture & operations docs
├── history/                 ← Per-repo JSON run logs
├── prompts/                 ← Cached agent instructions
└── covers/                  ← SVG cover images for Notion
```

## Code Style

- Python 3.11+, type annotations on every function
- `ruff` for linting and formatting (see `pyproject.toml`)
- Google-style docstrings on every public class and function
- `logging` instead of `print()`
- No bare `except:` — always catch specific exception types

## Reporting Issues

Please use the GitHub issue templates:
- **Bug report** — for unexpected failures
- **Feature request** — for new ideas

For security issues, see [SECURITY.md](SECURITY.md).
