.PHONY: install test lint format run train diagram docker-up docker-down

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check . --fix

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

train:
	curl -s -X POST http://localhost:8000/api/v1/train | python3 -m json.tool

diagram:
	python scripts/generate_diagram.py

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v
