.PHONY: install test lint format run docker-up docker-down clean help

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
