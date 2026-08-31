.PHONY: help install test test-cov test-fast lint lint-fix format typecheck check serve benchmark seed clean

PYTHON ?= python
PIP    ?= pip

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*##"}; {printf "  %-18s %s\n", $$1, $$2}'

install:  ## Install package and dev dependencies
	$(PIP) install -e ".[dev]"

test:  ## Run the full test suite
	pytest -q --tb=short

test-cov:  ## Run tests with coverage report
	pytest -q --tb=short --cov=app --cov-report=term-missing

test-fast:  ## Run tests, stopping at the first failure
	pytest -q --tb=short -x

lint:  ## Run ruff linter
	ruff check app/ tests/

lint-fix:  ## Run ruff linter and apply safe autofixes
	ruff check app/ tests/ --select E,F,W,I --ignore E501 --fix

format:  ## Auto-format with ruff
	ruff format app/ tests/

typecheck:  ## Run mypy static type checker
	mypy app/ --ignore-missing-imports

check: lint typecheck test  ## Run lint, typecheck, and tests (what CI runs)

serve:  ## Run the API locally with auto-reload
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

benchmark:  ## Run micro-benchmarks
	$(PYTHON) scripts/benchmark.py --runs 500

seed:  ## Seed development data
	$(PYTHON) scripts/seed_data.py --verbose

clean:  ## Remove compiled Python files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
