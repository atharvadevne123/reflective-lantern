# Contributing to Property-Sage

## Development setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern/Property-Sage
pip install -r requirements.txt
cp .env.example .env
```

## Running tests

```bash
pytest tests/ -v
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check . --fix
```

All PRs must pass `ruff check .` with zero errors and all existing tests.

## Commit convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `test:` — new or updated tests
- `docs:` — documentation only
- `chore:` — tooling, deps, config
- `ci:` — CI pipeline changes

## Pull requests

- Keep PRs focused on a single concern.
- Include tests for any new behaviour.
- Update `CHANGELOG.md` under `[Unreleased]`.
