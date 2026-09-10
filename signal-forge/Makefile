.PHONY: install test lint format run docker-build docker-up clean

install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio httpx ruff pre-commit

test:
	python -m pytest tests/ -v --tb=short

lint:
	ruff check . --select E,F,W,I --ignore E501

format:
	ruff format .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t signal-forge .

docker-up:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f test_signal_forge.db
