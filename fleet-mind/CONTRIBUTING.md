# Contributing to Fleet-Mind

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd fleet-mind
pip install -r requirements.txt
pre-commit install
```

## Running Tests

```bash
make test
```

## Code Style

- Python 3.11+, formatted with **ruff**
- All public functions must have Google-style docstrings
- All function signatures must have type annotations
- No bare `except:` — always catch specific exceptions and log them

## Pull Request Guidelines

1. Branch from `main` with a descriptive name: `feat/add-fuel-efficiency-feature`
2. One logical change per PR
3. Ensure `make lint` and `make test` pass before submitting
4. Update the relevant section of `CHANGELOG.md`

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): short description
fix(scope): short description
test(scope): short description
docs(scope): short description
chore(scope): short description
```
