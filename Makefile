.PHONY: smoke sanity ab test test-cov lint format docker-build docker-up demo publish clean

smoke:
	pytest tests/smoke -v -m smoke

sanity:
	pytest tests/sanity -v -m sanity

ab:
	pytest tests/ab -v -m ab

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=wavqwise --cov-report=html

lint:
	ruff check wavqwise/
	mypy wavqwise/ --ignore-missing-imports

format:
	black wavqwise/ tests/
	isort wavqwise/ tests/

docker-build:
	docker build -t wavqwise -f docker/Dockerfile .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

demo:
	python demos/demo_forecasting.py
	python demos/demo_anomaly_detection.py

publish:
	python -m build
	twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info __pycache__ .pytest_cache htmlcov/
