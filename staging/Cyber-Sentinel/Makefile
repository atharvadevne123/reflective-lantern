.PHONY: install test coverage check lint format run docker-build docker-up clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

coverage:
	pytest tests/ -q --cov=app --cov-report=html --cov-report=term

check: lint test

lint:
	ruff check . --select E,F,W,I --ignore E501
	ruff format --check .

format:
	ruff check . --select E,F,W,I --ignore E501 --fix
	ruff format .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t cyber-sentinel:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .coverage coverage.xml .pytest_cache .ruff_cache
