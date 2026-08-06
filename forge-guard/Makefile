.PHONY: install install-dev test test-cov lint format run docker-up docker-down \
        retrain diagram clean typecheck healthcheck seed

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt pytest pytest-asyncio httpx ruff mypy

test:
	pytest tests/ -v --tb=short -q 2>&1 | tail -80

test-cov:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=xml -q 2>&1 | tail -40

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

typecheck:
	mypy app/ --ignore-missing-imports --no-error-summary 2>&1 | tail -20 || true

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

retrain:
	python pipelines/retrain_dag.py manual

diagram:
	python scripts/generate_diagram.py

healthcheck:
	python scripts/healthcheck.py

seed:
	python scripts/seed_data.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f forge_guard.db test_forge_guard.db model.joblib metrics.json coverage.xml
