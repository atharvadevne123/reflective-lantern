# Contributing to Trade-Sage

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running tests

```bash
make test
```

## Code style

Ruff handles formatting and linting. Run `make lint-fix` before committing.

## Pull request process

1. Open a draft PR against `main`
2. Ensure all CI checks pass
3. Request a review
