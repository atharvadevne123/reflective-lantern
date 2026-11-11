# Contributing to Risk-Sense

## Development Setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern/risk-sense
pip install -r requirements.txt
pre-commit install
```

## Code Style

- Python 3.11+, type annotations on all public functions
- Google-style docstrings on all classes and public methods
- `ruff` for linting and formatting (run `make lint`)
- Max line length: 120

## Testing

- All new features require tests in `tests/`
- Run `make test` before committing
- Target: 80%+ coverage

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feature/my-change`
2. Make changes with atomic commits (`fix(scope): description`)
3. Run `make lint && make test`
4. Open a PR with a clear description of the change

## Commit Convention

```
type(scope): short description

Types: feat, fix, test, docs, refactor, ci, chore
```
