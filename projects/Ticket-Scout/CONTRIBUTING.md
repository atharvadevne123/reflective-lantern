# Contributing to Ticket-Scout

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running tests

```bash
make test
```

## Linting

```bash
make lint      # check only
make format    # auto-fix
```

## Pull requests

- One feature or fix per PR
- Include tests for new functionality
- `ruff check .` must pass with zero errors
- All existing tests must pass
