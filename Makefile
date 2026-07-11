.PHONY: install test lint format run train diagram docker-up docker-down \
        coverage typecheck clean help migrate check

## help: Show this help message
help:
	@grep -E '^## [a-z]' Makefile | sed 's/## /  make /'

## install: Install Python dependencies
install:
	pip install -r requirements.txt

## test: Run the full test suite
test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

## coverage: Run tests with HTML coverage report
coverage:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html:htmlcov -q
	@echo "Coverage report: htmlcov/index.html"

## typecheck: Run mypy type checks on the app package
typecheck:
	mypy app/ --ignore-missing-imports --no-error-summary

## lint: Check code style with ruff
lint:
	ruff check .
	ruff format --check .

## format: Auto-format and fix lint issues with ruff
format:
	ruff format .
	ruff check . --fix

## check: Run lint + typecheck together (CI equivalent)
check: lint typecheck

## run: Start the FastAPI dev server with hot-reload
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

## train: Trigger model training via the local API
train:
	curl -s -X POST http://localhost:8000/api/v1/train | python3 -m json.tool

## migrate: Initialise / migrate the database schema
migrate:
	python -c "from app.database import Base, engine; Base.metadata.create_all(bind=engine); print('DB tables created.')"

## diagram: Regenerate the system-architecture PNG
diagram:
	python scripts/generate_diagram.py

## docker-up: Build and start the Docker Compose stack
docker-up:
	docker-compose up --build -d

## docker-down: Tear down the Docker Compose stack and volumes
docker-down:
	docker-compose down -v

## clean: Remove Python cache files, coverage artefacts, and test DBs
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .mypy_cache .ruff_cache
	rm -f test_watt_guard.db watt_guard.db
	@echo "Clean complete."
