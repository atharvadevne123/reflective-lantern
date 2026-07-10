# Contributing to Volt-Cast

## Development Setup

```bash
git clone https://github.com/atharvadevne123/reflective-lantern
cd reflective-lantern
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

This project uses `ruff` for linting and formatting:

```bash
ruff check .
ruff format .
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Add tests for any new functionality
3. Ensure all tests pass and ruff shows no errors
4. Submit a pull request with a clear description
