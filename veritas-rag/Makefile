.PHONY: install test lint format run demo eval diagram clean help

help:
	@echo "Veritas-Rag targets: install test lint format run demo eval diagram clean"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

demo:
	python scripts/demo.py

eval:
	python scripts/run_eval.py

diagram:
	python scripts/generate_diagram.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
