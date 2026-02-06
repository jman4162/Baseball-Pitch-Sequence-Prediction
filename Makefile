.PHONY: install data train benchmark ablation mlflow test clean build docs docs-serve

install:
	pip install -e ".[all,dev]"

data:
	pitch-generate

train:
	pitch-train --model $(MODEL)

benchmark:
	pitch-benchmark

ablation:
	pitch-ablation --type $(TYPE)

mlflow:
	mlflow ui --backend-store-uri experiments

test:
	pytest tests/

build:
	python -m build

docs:
	mkdocs build

docs-serve:
	mkdocs serve

clean:
	rm -rf experiments/ __pycache__ .pytest_cache dist/ *.egg-info site/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
