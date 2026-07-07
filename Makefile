.PHONY: install test lint run diagram clean

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
	uvicorn app.main:app --reload --port 8000

diagram:
	python scripts/generate_diagram.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f realty_edge.db test_realty_edge.db model.joblib metrics.json
