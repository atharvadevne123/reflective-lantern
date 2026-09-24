# Contributing to Logistics-Flow

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

## Before you open a pull request

Run the same checks CI runs:

```bash
make lint     # ruff check .
make test     # pytest tests/
```

Both must pass. CI additionally enforces `ruff format --check`.

## Code style

- Ruff handles linting and formatting; line length is 100
- Type annotations on all public functions
- Google-style docstrings on modules and public functions
- Use `logger`, never `print`, in `app/`

## Tests

Every behavioural change needs a test. Place tests in the file matching the
module under test (`app/features.py` → `tests/test_features.py`). Prefer
`pytest.mark.parametrize` over near-duplicate test bodies. External calls and
the database are mocked or use the in-memory SQLite fixture from `conftest.py`.

## Commits

Use Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`.
Keep each commit to a single logical change.

## Adding a feature to the model

1. Add the transform to `app/features.py`
2. Append the column name to `FEATURE_COLS`
3. Add a test asserting the column exists and its invariants hold
4. Retrain so `metrics.json` reflects the new feature count
