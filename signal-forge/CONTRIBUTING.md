# Contributing to Signal-Forge

## Setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern/signal-forge
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx ruff pre-commit
pre-commit install
```

## Running Tests

```bash
make test
```

## Linting

```bash
make lint
make format
```

## Commit Style

Use conventional commits: `feat`, `fix`, `test`, `docs`, `chore`, `refactor`, `ci`.

## Pull Requests

- One logical change per PR
- All tests must pass
- Ruff lint must pass with zero errors
