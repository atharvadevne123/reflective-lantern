.PHONY: install test lint run build clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .
	ruff format --check .

lint-fix:
	ruff check . --fix
	ruff format .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

build:
	pip install build -q
	python -m build --wheel

clean:
	rm -rf dist/ build/ *.egg-info __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
