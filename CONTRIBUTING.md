# Contributing to Watt-Guard

## Setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern
pip install -r requirements.txt
cp .env.example .env   # fill in values
```

## Running Tests

```bash
make test          # run full pytest suite
make coverage      # run with HTML coverage report (htmlcov/index.html)
```

## Linting & Type Checking

```bash
make lint          # ruff check + ruff format --check
make format        # auto-fix style issues
make typecheck     # mypy on the app/ package
make check         # lint + typecheck together (mirrors CI)
```

## Database

```bash
make migrate       # create / update SQLite schema for local dev
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Write or update tests for your change
4. Ensure `make check` and `make test` pass with no new failures
5. Commit with [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(api): add batch predict endpoint`
   - `fix(model): prevent divide-by-zero in drift detector`
   - `docs(readme): update quick-start section`
6. Open a pull request against `main`

## Commit Message Format

```
<type>(<scope>): <short summary>

[optional body]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`

## Code Style

- **Python 3.11+** only — use modern syntax (`match`, `X | Y` unions, etc.)
- **Ruff** for linting and formatting (line length 120)
- **Google-style docstrings** on all public functions and classes
- **Type annotations** on every public function signature
- No bare `except` — always catch a specific exception type
- Prefer `logger.info/warning/error` over `print`

## Adding a New Endpoint

1. Define the request/response schema in `app/schemas.py`
2. Add the route handler in `app/main.py`
3. Write at least one happy-path and one error-case test in `tests/test_api.py`
4. Document the endpoint in `README.md`

## Adding a New ML Feature

1. Implement the transformer in `app/features.py` as an `sklearn` `BaseEstimator` + `TransformerMixin`
2. Add the transformer to the pipeline in `app/model.py::build_pipeline`
3. Update `tests/test_features.py` with a parametrized test

## Environment Variables

See `.env.example` for the full list. The minimum required for tests is:

```bash
DATABASE_URL=sqlite:///./test_watt_guard.db
```

## Docker

```bash
make docker-up    # build and start API + DB containers
make docker-down  # tear down
```
