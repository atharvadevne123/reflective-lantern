.PHONY: install test lint format run train clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .

format:
	ruff format .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

train:
	python -c "from app.features import make_synthetic_dataset; from app.model import train_models; X,yp,ys=make_synthetic_dataset(); m=train_models(X,yp,ys); print(m)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist build *.egg-info
