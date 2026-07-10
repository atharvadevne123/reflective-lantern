# Contributing to Temporal-Pulse

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository and clone your fork.
2. Install dependencies: `make install`
3. Copy the environment template: `cp .env.example .env`
4. Run the test suite: `make test`

## Development Workflow

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes with tests.
3. Run lint and type checks: `make lint typecheck`
4. Format your code: `make format`
5. Ensure all tests pass: `make test`
6. Open a pull request with a clear description.

## Commit Convention

We follow Conventional Commits:

- `feat:` new features
- `fix:` bug fixes
- `test:` test additions or changes
- `docs:` documentation only
- `refactor:` code restructuring without behaviour change
- `chore:` tooling, CI, dependencies

## Code Standards

- Python 3.11+, type annotations required for public functions
- Google-style docstrings on all public classes and functions
- All I/O paths must have error handling
- New endpoints require Pydantic validation and tests
- Keep functions under 40 lines; extract helpers when needed

## Testing

- Every new feature needs at least one happy-path test and one edge case.
- Use the fixtures in `tests/conftest.py` (in-memory SQLite, sample readings).
- Run `python -m pytest tests/ -v` before pushing.

## Questions?

Open an issue with the `question` label.
