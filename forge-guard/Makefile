.PHONY: install test lint format run docker-up docker-down retrain diagram clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short -q 2>&1 | tail -60

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

retrain:
	python pipelines/retrain_dag.py manual

diagram:
	python scripts/generate_diagram.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f forge_guard.db test_forge_guard.db model.joblib metrics.json
