# Contributing to Ticket-Oracle

## Setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd ticket-oracle
pip install -r requirements.txt
```

## Running tests

```bash
make test
```

## Linting

```bash
make lint    # check
make format  # auto-fix
```

## Commit convention

`type(N/total): short description` — e.g. `feat(3/60): add feature pipeline`.

Types: `feat`, `fix`, `test`, `ci`, `docs`, `chore`, `refactor`.

## Pull requests

- All PRs must pass ruff lint and pytest before review.
- Add tests for any new model feature or API endpoint.
- Keep PRs focused — one concern per PR.
