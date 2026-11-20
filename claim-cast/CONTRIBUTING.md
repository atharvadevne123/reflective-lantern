# Contributing to Claim-Cast

## Setup
```bash
pip install -r requirements.txt
make test
```

## Code Style
- Python 3.11+, ruff for linting (`make lint`)
- Type annotations on all public functions
- Google-style docstrings

## Tests
Run `make test`. All PRs require passing CI (ruff + pytest).

## Pull Requests
- One feature/fix per PR
- Add tests for new endpoints or model changes
- Update CHANGELOG.md
