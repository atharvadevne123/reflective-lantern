.PHONY: install test lint run docker-build docker-up clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short

lint:
	python -m ruff check . --select E,F,W,I --ignore E501

lint-fix:
	python -m ruff check . --select E,F,W,I --ignore E501 --fix

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t social-pulse:latest .

docker-up:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -f social_pulse.db test_social_pulse.db
