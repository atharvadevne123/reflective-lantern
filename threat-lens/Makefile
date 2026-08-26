.PHONY: install test lint format run train migrate docker clean help

help:
	@echo "Threat-Lens — available targets:"
	@echo "  install   Install Python dependencies"
	@echo "  test      Run the pytest suite"
	@echo "  lint      Run ruff checks"
	@echo "  format    Auto-fix lint issues with ruff"
	@echo "  train     Train the ensemble model from synthetic data"
	@echo "  migrate   Apply Alembic database migrations"
	@echo "  run       Start the FastAPI server on :8000"
	@echo "  docker    Build and start the docker-compose stack"
	@echo "  clean     Remove build artefacts and caches"

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	ruff check .

format:
	ruff check . --fix

train:
	python -c "from app.features import generate_synthetic_dataset; from app.model import train_model; X, y = generate_synthetic_dataset(); print(train_model(X, y)[1])"

migrate:
	alembic upgrade head

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker compose up --build

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache dist build *.egg-info
	find . -name "*.pyc" -delete
	rm -f test_threat_lens.db experiment_runs.jsonl
