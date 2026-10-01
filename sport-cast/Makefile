.PHONY: install test lint run clean diagram

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .
	ruff format --check .

fix:
	ruff check . --fix
	ruff format .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

diagram:
	python scripts/generate_diagram.py

clean:
	rm -f *.joblib *.json *.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v
