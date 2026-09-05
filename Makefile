.PHONY: install dev test run docker openapi clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	python -m pytest tests -q

run:
	uvicorn app.main:app --reload --port 8080

docker:
	docker build -t reservoirx-d:latest .
	docker run --rm -p 8080:8080 reservoirx-d:latest

openapi:
	python scripts/export_openapi.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache openapi.json
