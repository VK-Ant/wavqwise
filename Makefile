.PHONY: smoke sanity ab integration test test-cov lint format docker-build docker-up demo publish clean

smoke:
	pytest tests/smoke -v -m smoke

sanity:
	pytest tests/sanity -v -m sanity

ab:
	pytest tests/ab -v -m ab

integration:
	pytest tests/integration -v -m integration

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=wavqwise --cov-report=html --cov-report=term-missing

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
	python demos/demo_eeg_analysis.py

demo-real:
	python demos/demo_eeg_real_data.py
	python demos/demo_trading_real_data.py

publish:
	python -m build
	twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info __pycache__ .pytest_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
