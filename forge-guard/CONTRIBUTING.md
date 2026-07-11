# Contributing to Forge-Guard

## Getting started

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern/forge-guard
pip install -r requirements.txt
cp .env.example .env
```

## Running tests

```bash
make test
```

All tests must pass before opening a PR.

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
make format   # auto-fix
make lint     # check only (CI gate)
```

## Commit convention

```
type(N/60): short description
```

Types: `feat`, `fix`, `test`, `ci`, `docs`, `chore`, `refactor`.

## Pull request checklist

- [ ] `make lint` exits 0
- [ ] `make test` exits 0 with no new failures
- [ ] New features include at least one test
- [ ] `.env.example` updated if new env vars are introduced
