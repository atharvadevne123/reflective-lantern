.PHONY: help install test lint format benchmark seed clean

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

lint:  ## Run ruff linter
	ruff check app/ tests/

format:  ## Auto-format with ruff
	ruff format app/ tests/

typecheck:  ## Run mypy static type checker
	mypy app/ --ignore-missing-imports

benchmark:  ## Run micro-benchmarks
	$(PYTHON) scripts/benchmark.py --runs 500

seed:  ## Seed development data
	$(PYTHON) scripts/seed_data.py --verbose

clean:  ## Remove compiled Python files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
