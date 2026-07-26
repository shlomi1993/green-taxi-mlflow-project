.PHONY: install test lint mlflow run-baseline run-shifted clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check src flows tests

mlflow:
	mlflow server --workers 1 --port 5001 --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --default-artifact-root mlflow_tracking/mlruns

run-baseline:
	python flows/monitoring_flow.py run --batch-path data/raw/green_tripdata_2020-01.parquet

run-shifted:
	python flows/monitoring_flow.py run --ref-path data/raw/green_tripdata_2020-01.parquet --batch-path data/raw/green_tripdata_2020-04.parquet

clean:
	rm -rf .metaflow mlflow_tracking mlruns .pytest_cache
