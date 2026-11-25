# Contributing to Support-Sage

## Setup

```bash
git clone https://github.com/atharvadevne123/Support-Sage
cd Support-Sage
pip install -r requirements.txt
cp .env.example .env
make train
make run
```

## Development

- Run `make lint` before committing
- Run `make test` to verify all tests pass
- Follow [Conventional Commits](https://conventionalcommits.org) for commit messages
- Open a pull request against `main`

## Code Style

Ruff is configured in `pyproject.toml`. Line length is 120.
