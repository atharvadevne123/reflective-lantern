.PHONY: install test lint format run docker-up docker-down clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .

format:
	ruff format .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

diagram:
	python scripts/generate_diagram.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -f volt_cast.db test_volt_cast.db volt_cast_model.joblib volt_cast_metrics.json

retrain:
	python -c "from pipelines.retrain_dag import run_retraining_pipeline; run_retraining_pipeline()"
