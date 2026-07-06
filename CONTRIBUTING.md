# Contributing to Realty-Edge

## Setup

```bash
git clone https://github.com/atharvadevne123/Realty-Edge
cd Realty-Edge
pip install -r requirements.txt
cp .env.example .env
```

## Running tests

```bash
make test
```

## Lint

```bash
make lint-fix
```

## Submitting changes

1. Fork the repo and create a feature branch
2. Write tests for any new logic
3. Run `make lint` and `make test` — both must pass
4. Open a pull request with a clear description
