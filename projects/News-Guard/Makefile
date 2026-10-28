.PHONY: install test lint run docker-up docker-down diagram

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .
	ruff check . --fix

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

diagram:
	python scripts/generate_diagram.py
