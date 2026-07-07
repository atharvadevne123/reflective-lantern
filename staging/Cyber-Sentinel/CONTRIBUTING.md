# Contributing to Cyber-Sentinel

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/atharvadevne123/Cyber-Sentinel.git
cd Cyber-Sentinel
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
make test
```

## Code Style

We use `ruff` for linting and formatting:

```bash
make lint    # check
make format  # auto-fix
```

## Submitting a Pull Request

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit your changes following [Conventional Commits](https://www.conventionalcommits.org/)
4. Open a pull request against `main`

## Commit Message Format

```
type(scope): short description

feat:     new feature
fix:      bug fix
docs:     documentation only
test:     add or update tests
ci:       CI/CD changes
refactor: code refactoring
chore:    maintenance
```
