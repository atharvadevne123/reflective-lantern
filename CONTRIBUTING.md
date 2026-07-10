# Contributing to Watt-Guard

## Setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern
git checkout innovation/watt-guard
pip install -r requirements.txt
```

## Running Tests

```bash
make test
```

## Linting

```bash
make lint
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit with conventional commits: `feat(api): add batch predict endpoint`
4. Open a pull request against `innovation/watt-guard`

## Code Style

- Python 3.11+
- Ruff for linting and formatting (line length 120)
- Google-style docstrings
- Type annotations on all public functions
