.PHONY: install test lint format typecheck coverage watch run docker clean diagram

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	python -m ruff check .

format:
	python -m ruff format .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

diagram:
	python scripts/generate_diagram.py

clean:
	rm -rf __pycache__ .pytest_cache *.joblib metrics.json *.db dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

smoke:
	python scripts/smoke_test.py --base-url $${BASE_URL:-http://localhost:8000}

benchmark:
	python scripts/benchmark.py

typecheck:
	python -m mypy app

coverage:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

watch:
	pytest-watch tests/ -- -v --tb=short

migrate:
	alembic upgrade head
