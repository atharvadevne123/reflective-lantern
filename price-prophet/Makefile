.PHONY: install test lint run docker-build docker-up clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	ruff check . --select E,F,W,I --ignore E501
	ruff check . --select E,F,W,I --ignore E501 --fix

run:
	uvicorn app.api.server:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t price-prophet:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
