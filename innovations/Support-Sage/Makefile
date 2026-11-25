.PHONY: install test lint run train clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .
	ruff check . --fix

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

train:
	python -c "from app.features import generate_synthetic_data; from app.model import train_model; train_model(generate_synthetic_data(3000))"

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info
