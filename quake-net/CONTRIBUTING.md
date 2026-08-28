# Contributing to Quake-Net

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env
```

## Before opening a pull request

```bash
make lint      # ruff check
make format    # ruff format + autofix
make test      # pytest with coverage
```

All three must pass. CI runs the same commands on Python 3.11 and 3.12.

## Code standards

- **Type annotations** on every public function signature.
- **Google-style docstrings** on public functions and classes. Private helpers need one
  only when the behaviour is non-obvious.
- **Structured logging** via the module-level `logger`; never `print` in `app/`.
- **Line length 100**, enforced by ruff.
- New behaviour ships with tests. Bug fixes ship with a regression test that fails without
  the fix.

## Adding a feature transform

Feature transforms live in `app/features.py` and must implement the sklearn
`BaseEstimator` / `TransformerMixin` contract so they serialise with the model:

1. Add the transformer class with `fit` and `transform`.
2. Register it in `build_feature_pipeline()` at the right position — ordering matters, as
   `DropCategoricalColumns` and `InfinityNaNFixer` must stay ahead of `StandardScaler`.
3. Add tests to `tests/test_features.py` covering row-count preservation, absence of
   NaN/inf in the output, and the specific columns produced.

Retrain after any pipeline change: a stale `model.joblib` carries the old feature layout
and will fail at predict time with a shape mismatch.

## Adding an endpoint

1. Define Pydantic request and response models with field constraints and descriptions.
2. Mount the route under `/api/v1` with a `tags` value and a docstring — `tests/test_api.py`
   asserts that every route is versioned and documented.
3. Add tests covering the success path, a validation rejection, and the empty-data path.

## Commit messages

Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`, `refactor:`,
`perf:`. One logical change per commit.

## Reporting bugs

Include the request payload, the full response, the `X-Correlation-ID` header, and the
relevant log lines. For model quality issues, include `metrics.json` and the output of
`GET /api/v1/drift`.
