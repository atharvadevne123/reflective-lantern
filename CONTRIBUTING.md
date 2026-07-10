# Contributing to Traffic-Pulse

Thanks for your interest in contributing!

## Getting started

1. Fork the repository and clone your fork.
2. Install dependencies: `make install`
3. Run the test suite: `make test`
4. Run the linter: `make lint`

## Development workflow

- Create a feature branch from `main`: `git checkout -b feat/my-feature`
- Keep changes focused — one logical change per pull request.
- Add or update tests for any behaviour change.
- Ensure `ruff check .` and `pytest` pass before pushing.

## Commit style

Use conventional commit prefixes:

- `feat:` new functionality
- `fix:` bug fixes
- `test:` test-only changes
- `docs:` documentation
- `chore:` tooling / maintenance
- `ci:` CI configuration

## Code style

- Python 3.11+, full type annotations on public functions.
- Google-style docstrings.
- `ruff` handles formatting and linting (config in `pyproject.toml`).

## Reporting issues

Open a GitHub issue with a minimal reproduction, expected vs actual
behaviour, and your environment details.
