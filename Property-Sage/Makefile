.PHONY: install test lint run retrain clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check . --fix
	ruff check .

run:
	DATABASE_URL=sqlite:///./property_sage.db uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

retrain:
	PYTHONPATH=. python pipelines/retrain_dag.py

diagram:
	PYTHONPATH=. python scripts/generate_diagram.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -f property_sage.db test_property_sage.db; \
	rm -rf models/ dist/ build/ *.egg-info
