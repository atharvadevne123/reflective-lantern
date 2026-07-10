.PHONY: install test lint run docker-build docker-up clean format typecheck

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff mypy

test:
	python -m pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

lint:
	ruff check . --select E,F,W,I --ignore E501

format:
	ruff check . --select E,F,W,I --ignore E501 --fix
	ruff format .

typecheck:
	mypy app/ --ignore-missing-imports --no-strict-optional

run:
	uvicorn app.main:app --reload --port 8000

docker-build:
	docker build -t temporal-pulse:latest .

docker-up:
	docker compose up -d

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml
