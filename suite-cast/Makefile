.PHONY: install test test-cov test-fast lint format typecheck run docker-up docker-down clean help check-all

help:
	@echo "Suite-Cast development targets:"
	@echo "  install     Install dependencies"
	@echo "  test        Run pytest suite"
	@echo "  lint        Run ruff checks"
	@echo "  format      Auto-format with ruff"
	@echo "  run         Start dev server on :8000"
	@echo "  docker-up   Start API + PostgreSQL via compose"
	@echo "  docker-down Stop compose stack"
	@echo "  clean       Remove artifacts and caches"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=app --cov-report=term-missing -q 2>&1 | tail -40

test-fast:
	pytest tests/ -x -q --tb=short 2>&1 | tail -40

typecheck:
	mypy app/ --ignore-missing-imports --no-error-summary 2>&1 | tail -20 || true

check-all: lint typecheck test-cov

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -f model.joblib lgbm_model.joblib metrics.json test_suite_cast.db suite_cast.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

train:
	python scripts/train.py --samples 2000

diagram:
	python scripts/generate_diagram.py

demo:
	python scripts/demo_request.py
