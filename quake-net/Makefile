.PHONY: install test lint format run docker-up docker-down train clean

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov httpx ruff

test:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing 2>&1 | tail -60

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

train:
	python3 -c "from app.model import train_model; _, m = train_model(); print('Metrics:', m)"

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f model.joblib metrics.json reference_dist.json
	rm -f quake_net.db test_quake_net.db
