.PHONY: install lint test test-fast build clean seed run

install:
	pip install -e ".[dev]"

lint:
	ruff check ticket-oracle/

format:
	ruff format ticket-oracle/

test:
	python -m pytest ticket-oracle/tests/ -v --tb=short

test-fast:
	python -m pytest ticket-oracle/tests/ -v --tb=short -x -q \
		--ignore=ticket-oracle/tests/test_model.py

build:
	python -m build --sdist --wheel ticket-oracle/

clean:
	rm -rf ticket-oracle/dist/ ticket-oracle/*.egg-info ticket-oracle/__pycache__ \
		ticket-oracle/app/__pycache__ ticket-oracle/tests/__pycache__ \
		ticket-oracle/models/ ticket_oracle.db .pytest_cache

seed:
	cd ticket-oracle && python scripts/seed_db.py --n 500

run:
	cd ticket-oracle && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:
	cd ticket-oracle && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

diagram:
	cd ticket-oracle && python scripts/generate_diagram.py

benchmark:
	cd ticket-oracle && python scripts/benchmark.py --n 50
