.PHONY: install test lint fmt run docker clean diagram retrain

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .

fmt:
	ruff check . --fix
	ruff format .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker:
	docker-compose up --build

diagram:
	python scripts/generate_diagram.py

retrain:
	python -c "from pipelines.retrain_dag import run_retrain_pipeline; run_retrain_pipeline()"

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache dist build *.egg-info
	find . -name "*.pyc" -delete
	rm -f test_cyber_guard.db

migrate:
	alembic upgrade head

migrate-down:
	alembic downgrade -1

seed:
	python scripts/seed_data.py

seed-drift:
	python scripts/seed_data.py --drift
