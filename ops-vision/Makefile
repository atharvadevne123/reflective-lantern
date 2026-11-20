.PHONY: help install install-dev test test-cov lint format typecheck run docker-build docker-up docker-down clean

help:
	@echo "Ops-Vision — available targets:"
	@echo "  install       Install runtime dependencies"
	@echo "  install-dev   Install runtime + development dependencies"
	@echo "  test          Run the pytest suite"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  lint          Run ruff linter"
	@echo "  format        Auto-format with ruff"
	@echo "  typecheck     Run mypy static type checks"
	@echo "  run           Start the API with uvicorn (reload enabled)"
	@echo "  docker-build  Build the Docker image"
	@echo "  docker-up     Start API + PostgreSQL via docker-compose"
	@echo "  docker-down   Stop and remove compose services"
	@echo "  clean         Remove caches and build artifacts"

install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-cov pytest-asyncio httpx ruff mypy pre-commit
	pre-commit install

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

lint:
	python -m ruff check . --select E,F,W,I --ignore E501

format:
	python -m ruff check . --select E,F,W,I --ignore E501 --fix
	python -m ruff format .

typecheck:
	python -m mypy app/ --ignore-missing-imports --no-strict-optional

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t ops-vision:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
