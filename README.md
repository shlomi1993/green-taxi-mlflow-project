# 🚕 Green Taxi MLflow Project

<img width="1672" height="941" alt="green-taxi-project-overview" src="https://github.com/user-attachments/assets/c4fd34e2-8dcc-4dcc-b5f6-9c9ecc021073" />

Production-oriented batch monitoring and retraining workflow for NYC green taxi tip prediction. The system watches incoming batches, validates data quality, evaluates the registered champion model, retrains when performance degrades, and promotes a stronger candidate through MLflow Model Registry.

## ✨ What This Repo Demonstrates

- Metaflow orchestration for repeatable batch workflows
- MLflow tracking, artifact logging, dataset lineage, and model registry aliases
- Hard data-quality gates plus NannyML drift checks
- Champion/candidate evaluation with explicit promotion criteria
- Reproducible local execution, tests, and operational runbooks

## 🏗️ Architecture

```text
raw TLC batch
  -> integrity gate
  -> feature engineering
  -> champion evaluation
  -> retrain decision
  -> candidate training
  -> promotion gate
  -> champion alias update
```

The workflow is implemented in `flows/monitoring_flow.py`. Reusable data, validation, feature, model, and registry logic lives in `src/taxi_tip_ops/pipeline.py`.

For design rationale, operating boundaries, failure handling, and the production
roadmap, see [`docs/design.md`](docs/design.md).

## 📁 Repository Layout

```text
.
|-- configs/                  # Runtime defaults and quality gate thresholds
|-- data/raw/                 # Local TLC parquet files, ignored by git
|-- docs/                     # Architecture and operating notes
|-- flows/                    # Metaflow entrypoints
|-- scripts/                  # Local utility scripts
|-- src/taxi_tip_ops/         # Reusable pipeline package
|-- tests/                    # Unit and flow-level tests
|-- environment.yml           # Conda environment
|-- pyproject.toml            # Python package and test configuration
`-- Makefile                  # Common local commands
```

## 🚀 Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate taxi-tip-ops
python -m pip install -e .
```

Download the sample operating batches:

```bash
./scripts/download_tlc_data.sh
```

Start MLflow in a separate terminal:

```bash
make mlflow
```

Bootstrap and evaluate the champion:

```bash
make run-baseline
```

Evaluate a shifted batch and trigger retraining when the gate requires it:

```bash
make run-shifted
```

Open the MLflow UI at `http://localhost:5001` and inspect experiment `taxi_tip_monitoring`.

## 🛡️ Model Governance

A candidate is promoted only when all gate conditions pass:

- The batch passed hard integrity checks.
- Evaluation metrics and dataset lineage were logged.
- Candidate batch RMSE improves on the champion by at least 1 percent.
- Candidate reference RMSE does not regress by more than 5 percent.

Promotion updates the MLflow `@champion` alias on `green_taxi_tip_model`. Rejected candidates are still registered with validation tags so there is an audit trail.

## 🧪 Testing

```bash
make test
```

The tests use synthetic taxi batches and local file-based tracking where possible, so they can run without the MLflow server for most validation paths.

## ⚙️ Operational Notes

- Local tracking state is written to `mlflow_tracking/` and ignored by git.
- Raw TLC parquet files stay in `data/raw/` and are ignored by git.
- Thresholds are documented in `configs/model_monitoring.yaml`; the workflow exposes promotion and tracking parameters at runtime.
- Batch prediction artifacts are logged to MLflow as `predictions.parquet` for post-run inspection.
