.PHONY: install test lint run train diagram docker-up docker-down migrate coverage

install:
	pip install -r requirements.txt
	pip install pytest httpx ruff

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .
	ruff format --check .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

train:
	python -c "from app.model import train_model; train_model()"

diagram:
	python scripts/generate_diagram.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

migrate:
	alembic upgrade head

coverage:
	pytest tests/ --cov --cov-report=term-missing
