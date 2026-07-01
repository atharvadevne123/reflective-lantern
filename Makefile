.PHONY: install install-dev test lint format type-check clean help

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

help:
	@echo "Available targets:"
	@echo "  install      Install runtime dependencies"
	@echo "  install-dev  Install dev dependencies (includes runtime)"
	@echo "  test         Run pytest with coverage"
	@echo "  lint         Run ruff linter"
	@echo "  format       Run ruff formatter"
	@echo "  type-check   Run mypy type checker"
	@echo "  clean        Remove build artifacts and caches"

install:
	$(PIP) install -e . -q

install-dev:
	$(PIP) install -e ".[dev]" -q
	pre-commit install

test:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=config --cov=scripts --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check . --select E,F,W,I --ignore E501

format:
	$(PYTHON) -m ruff format .

type-check:
	$(PYTHON) -m mypy config/ scripts/ --ignore-missing-imports

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

notion-update:
	$(PYTHON) scripts/notion_portfolio_update.py

notion-update-descriptions:
	$(PYTHON) scripts/notion_portfolio_update.py --descriptions

health-check:
	$(PYTHON) scripts/health_check.py

weekly-summary:
	$(PYTHON) scripts/generate_weekly_summary.py

validate-history:
	$(PYTHON) scripts/validate_history.py
