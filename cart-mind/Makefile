.PHONY: install test lint format run build clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short 2>&1 | tail -80

test-fast:
	python -m pytest tests/ -q --tb=line --ignore=tests/test_pipeline.py

train:
	python -m scripts.train

diagram:
	python scripts/generate_diagram.py

lint:
	python -m ruff check .

format:
	python -m ruff format .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

build:
	pip install build && python -m build --wheel --sdist

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info
