.PHONY: install install-dev test lint format type-check clean help \
        summarize summarize-json ci-status all-checks report-daily report-weekly cleanup \
        notion-update notion-update-descriptions health-check weekly-summary validate-history \
        validate-json clean-cache lint-fix coverage test-fast

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
	@echo "  summarize    Print history summary table"
	@echo "  summarize-json Print history summary as JSON"
	@echo "  ci-status    Check CI status across all repos"
	@echo "  all-checks   Run all local health checks"
	@echo "  report-daily Generate today's Markdown report"
	@echo "  report-weekly Generate weekly Markdown report"
	@echo "  cleanup      Preview old history entries to remove (dry-run)"
	@echo "  validate-json  Validate history and output JSON results"
	@echo "  clean-cache  Remove __pycache__ and .pyc files"
	@echo "  lint-fix     Auto-fix lint issues with ruff"
	@echo "  coverage     Run tests with HTML coverage report"
	@echo "  test-fast    Quick test run (quiet, stop on first failure)"

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

summarize:
	$(PYTHON) scripts/summarize_history.py

summarize-json:
	$(PYTHON) scripts/summarize_history.py --json

ci-status:
	$(PYTHON) scripts/check_ci_status.py

all-checks:
	$(PYTHON) scripts/run_all_checks.py

report-daily:
	$(PYTHON) scripts/report_generator.py --mode daily

report-weekly:
	$(PYTHON) scripts/report_generator.py --mode weekly

cleanup:
	$(PYTHON) scripts/cleanup.py --dry-run

validate-json:
	$(PYTHON) scripts/validate_history.py --json

clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache

lint-fix:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff check . --unsafe-fixes || true

coverage:
	$(PYTHON) -m pytest tests/ --cov=config --cov=scripts --cov-report=html --cov-report=term-missing

test-fast:
	$(PYTHON) -m pytest tests/ -q --tb=no -x
