# Contributing to Social-Pulse

Thank you for your interest in contributing!

## Development Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

## Running Tests

```bash
make test
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
make lint
make lint-fix
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Write tests for any new functionality.
3. Ensure all tests pass and lint is clean.
4. Submit a pull request with a clear description of the change.

## Reporting Issues

Open a GitHub issue with a minimal reproducible example.
