# Contributing to Price-Prophet

Thank you for considering a contribution! Please follow these guidelines.

## Development Setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern.git
cd reflective-lantern/price-prophet
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

## Code Standards

- All Python files must pass `ruff check --select E,F,W,I --ignore E501`
- Type annotations required on all public functions
- Google-style docstrings on all classes and public methods
- No bare `except:` clauses — use `except Exception as e:`
- Replace `print()` with `logging.getLogger(__name__)` calls

## Testing

Run the full test suite before submitting a pull request:

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

All new functionality must include tests. Aim for ≥80% coverage on new modules.

## Pull Request Process

1. Fork and create a feature branch from `main`
2. Make your changes with clear, atomic commits
3. Ensure CI passes (lint + tests)
4. Open a pull request with a clear description of the change
5. Address review feedback promptly

## Commit Message Format

```
type(scope): short description

type: feat | fix | docs | test | refactor | chore | ci
```

## Reporting Issues

Open a GitHub issue with steps to reproduce, expected behaviour, and actual behaviour.
