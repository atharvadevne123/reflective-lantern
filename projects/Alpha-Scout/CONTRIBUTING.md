# Contributing to Alpha-Scout

## Setup

```bash
git clone https://github.com/atharvadevne123/Alpha-Scout
cd Alpha-Scout
pip install -r requirements.txt
cp .env.example .env
```

## Running Tests

```bash
make test
```

## Code Style

This project uses `ruff` for linting:

```bash
make lint
```

## Pull Request Guidelines

- Branch from `main`
- Write tests for new features
- Keep commits focused (one logical change per commit)
- Reference any related issues

## Architecture

See `screenshots/architecture.svg` for the system diagram.
