# Contributing to Threat-Lens

## Getting set up

```bash
pip install -r requirements.txt
pre-commit install
make train
make test
```

## Before you open a pull request

Run the same checks CI runs:

```bash
make lint     # ruff check .
make test     # pytest
```

Both must pass. `make format` applies ruff's autofixes if lint complains.

## Conventions

- **Commits** follow `type(scope): summary` — `feat`, `fix`, `test`, `docs`, `chore`, `ci`.
- **Type annotations** on every public function.
- **Docstrings** in Google style for anything exported from a module.
- **Logging** through the module-level `logger`, never `print`.
- `X` / `y` are the accepted names for feature matrices and label vectors; ruff's
  `N803`/`N806` are disabled for this reason.

## Adding a feature to the model

1. Append the feature name to `FEATURE_NAMES` in `app/features.py`.
2. Compute it in `NetworkFeatureEngineer._engineer_one`, keeping the return list in the
   same order as `FEATURE_NAMES`.
3. Add a test in `tests/test_features.py` asserting the computed value.
4. Retrain — a stale `model.joblib` expects the old feature count and will fail at
   inference.

## Adding threat intelligence

Entries live in `THREAT_INTEL_CORPUS` in `app/rag_retriever.py`. Each needs a stable
`id` (CVE or MITRE technique ID) and a `text` blurb that mentions the observable flow
characteristics — the retriever matches on those terms.

## Testing notes

- `tests/conftest.py` provisions an isolated SQLite database per session.
- Keep training sets small in tests; the fixtures use 300–500 samples deliberately.
- Do not set `n_jobs=-1` on both `cross_val_score` and a base estimator — nesting the
  two deadlocks joblib on constrained runners.
