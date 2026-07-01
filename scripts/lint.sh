#!/usr/bin/env bash
# Run all linters in sequence. Exits non-zero if any check fails.
set -euo pipefail

PYTHON=${PYTHON:-python3}
FAILED=0

echo "==> ruff lint"
$PYTHON -m ruff check . --select E,F,W,I --ignore E501 || FAILED=1

echo "==> ruff format check"
$PYTHON -m ruff format --check . || FAILED=1

echo "==> mypy type check"
$PYTHON -m mypy config/ scripts/ --ignore-missing-imports --no-error-summary 2>&1 | tail -5 || FAILED=1

if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "All checks passed."
else
    echo ""
    echo "One or more checks failed."
    exit 1
fi
