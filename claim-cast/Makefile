.PHONY: install test lint run docker-up docker-down clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	python -m ruff check . --fix
	python -m ruff check .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f claim_cast_model.joblib claim_cast_anomaly.joblib claim_cast_metrics.json claim_cast.db test_claim_cast.db
