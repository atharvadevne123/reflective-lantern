.PHONY: install test lint format run docker-up docker-down clean diagram health typecheck test-fast

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

diagram:
	python scripts/generate_diagram.py

health:
	python scripts/health_check.py

typecheck:
	python -m mypy app/ --ignore-missing-imports --no-error-summary 2>&1 | tail -20

test-fast:
	pytest tests/ -v --tb=short -x -q 2>&1 | tail -40

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -f *.db test_*.db model.joblib metrics.json
