.PHONY: install test lint format run docker-build docker-up clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

lint:
	python -m ruff check . --select E,F,W,I --ignore E501
	python -m ruff format --check .

format:
	python -m ruff check . --select E,F,W,I --ignore E501 --fix
	python -m ruff format .

typecheck:
	python -m mypy app/ --ignore-missing-imports --no-error-summary

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t fleet-mind:latest .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml test_fleet_mind.db
