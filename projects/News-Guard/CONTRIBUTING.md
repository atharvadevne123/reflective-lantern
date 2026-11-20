# Contributing to News-Guard

## Development setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern -b project/news-guard
cd reflective-lantern/projects/News-Guard
pip install -r requirements.txt
cp .env.example .env
make test
```

## Commit convention

`type(N/60): short description`

Types: `feat`, `fix`, `test`, `docs`, `ci`, `chore`, `refactor`

## Pull requests

1. Branch from `main`
2. Run `make lint test` before pushing
3. One PR per feature or fix
