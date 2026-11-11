.PHONY: install test lint run docker-build clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	ruff check . --select E,F,W,I --ignore E501

lint-fix:
	ruff check . --select E,F,W,I --ignore E501 --fix

run:
	DATABASE_URL="sqlite:///./risk_sense.db" uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t risk-sense:latest .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f risk_sense.db test_risk_sense.db
	rm -f /tmp/risk_sense_model.pkl /tmp/risk_sense_pipeline.pkl
