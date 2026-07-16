# Contributing to Suite-Cast

Thanks for your interest in improving Suite-Cast!

## Getting started

1. Fork and clone the repository.
2. `make install` to set up dependencies.
3. `pre-commit install` to enable lint hooks.
4. Create a feature branch: `git checkout -b feat/my-change`.

## Development workflow

- **Lint before pushing**: `make lint` must exit clean; `make format` autofixes most issues.
- **Tests are required**: every behavioural change needs a test. Run `make test`.
- **Type annotations**: all public functions carry full annotations.
- **Docstrings**: Google style on modules, classes, and public functions.

## Commit conventions

Use conventional commit prefixes: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`, `refactor:`.

## Pull requests

- Keep PRs focused — one logical change per PR.
- Describe *why*, not just *what*.
- CI (ruff + pytest) must pass before review.

## Reporting issues

Open a GitHub issue with reproduction steps, expected vs actual behaviour, and environment details.
