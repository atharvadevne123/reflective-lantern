#!/usr/bin/env bash
# Bootstrap a development environment for Reflective Lantern.
# Run: bash scripts/setup.sh
set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "==> Checking Python version..."
$PYTHON --version

echo "==> Creating virtual environment (.venv)..."
$PYTHON -m venv .venv

echo "==> Activating virtual environment..."
# shellcheck source=/dev/null
source .venv/bin/activate

echo "==> Installing dev dependencies..."
pip install --upgrade pip -q
pip install -e ".[dev]" -q

echo "==> Installing pre-commit hooks..."
pre-commit install

echo "==> Checking for .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "    .env created from .env.example — fill in your values."
else
    echo "    .env already exists, skipping."
fi

echo ""
echo "Setup complete. Activate the venv with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run:"
echo "  make test    # run pytest"
echo "  make lint    # run ruff"
