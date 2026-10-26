.PHONY: install test lint format run docker-up docker-down retrain

install:
	pip install -r requirements.txt

test:
	DATABASE_URL=sqlite:///./test.db MODEL_DIR=./test_models pytest tests/ -v --tb=short

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

retrain:
	python pipelines/retrain_dag.py

diagram:
	python scripts/generate_diagram.py
