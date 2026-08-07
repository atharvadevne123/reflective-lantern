# Contributing to Energy-Seer

## Setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd energy_seer
pip install -r requirements.txt
```

## Running Tests

```bash
make test
```

## Code Style

This project uses `ruff` for linting and formatting:

```bash
make lint    # check
make format  # auto-fix
```

## Pull Request Guidelines

- One feature or fix per PR
- All tests must pass
- Add tests for new features
- Keep commits atomic with descriptive messages
