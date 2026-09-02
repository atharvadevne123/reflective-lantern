# Contributing to Cyber-Guard

## Development Setup

```bash
cd Cyber-Guard
pip install -r requirements.txt
pre-commit install
cp .env.example .env
```

## Before Opening a Pull Request

All three must pass locally:

```bash
make lint    # ruff check . — must exit 0
make test    # pytest tests/ — all tests must pass
make run     # API must start and serve /api/v1/health
```

## Code Style

- **Formatting**: `ruff` with a 100-character line limit (`make fmt`).
- **Type hints**: required on all public functions.
- **Docstrings**: Google style on all public functions and classes.
- **Naming**: `X` / `y` are permitted for feature matrices and targets
  (sklearn convention); everything else is `snake_case`.

## Testing

- Every new feature needs a test.
- Tests must be isolated — the `db_session` fixture truncates all tables on
  teardown, so never rely on rows left behind by another test.
- External calls must be mocked; the suite runs offline in CI.
- Use `@pytest.mark.parametrize` for input variations rather than copy-pasting
  near-identical test bodies.

## Commit Messages

Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`.

## Adding a New Feature to the Model

1. Add the derivation to `NetworkFeatureEngineer.transform` in `app/features.py`.
2. Append the column name to `FEATURE_NAMES` — the order must match.
3. Update `n_features` expectations in `tests/test_features.py` and
   `tests/test_model.py`.
4. Retrain: `make retrain`.
