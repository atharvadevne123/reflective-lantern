.PHONY: install test lint run clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .
	ruff check . --fix

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f model.joblib metrics.json alpha_scout.db

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v
