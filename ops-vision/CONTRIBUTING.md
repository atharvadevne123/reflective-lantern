# Contributing to Ops-Vision

Thanks for your interest in improving Ops-Vision.

## Development setup

```bash
make install-dev   # installs runtime + dev deps and registers pre-commit hooks
cp .env.example .env
make test
```

## Before opening a pull request

Both of these must pass:

```bash
make lint          # ruff, selecting E,F,W,I
make test          # full pytest suite
```

`make format` applies the autofixes if lint complains.

## Code standards

- **Type annotations** on every function signature, including return types.
- **Google-style docstrings** on every module, class, and public function.
  Include `Args:`, `Returns:`, and `Raises:` where they apply.
- **Logging, not printing.** Use `logging.getLogger(__name__)`. Never `print()`
  in library code.
- **No bare `except:`.** Catch `Exception` and log with `logger.exception(...)`
  so the traceback survives.
- **No secrets in source.** Read configuration from `app.config.get_settings()`,
  and document any new variable in `.env.example`.
- **Functions stay under ~40 lines.** Extract named helpers past that.

## Testing expectations

- Every new endpoint needs a happy-path test plus validation-boundary tests.
- Every new feature transformer needs a correctness test and an edge-case test
  (zero, negative, and empty inputs at minimum).
- Use `@pytest.mark.parametrize` for boundary tables rather than copy-pasting
  near-identical test bodies.
- Tests must not require a running database. The suite uses SQLite with
  per-test transaction rollback; keep it that way.

## Commit messages

Conventional-commit prefixes: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`,
`chore:`, `ci:`, `perf:`. Keep the subject line under 72 characters and write it
in the imperative mood.

## Reporting issues

Include the Ops-Vision version, Python version, a minimal reproduction, and the
full traceback. If the issue involves a prediction, include the input payload
with any sensitive service names redacted.
