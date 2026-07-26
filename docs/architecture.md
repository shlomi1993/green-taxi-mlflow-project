# Architecture

## Objective

Keep a taxi tip prediction model reliable as demand patterns change. The workflow treats every new batch as an operational event: validate it, score the current champion, decide whether retraining is justified, and promote only if the candidate is both better and stable.

## Components

`green_taxi_flow.py` owns orchestration. Each Metaflow step has one clear responsibility and records its decision state in MLflow.

`src/pipeline.py` owns reusable logic: loading parquet or CSV data, integrity checks, NannyML drift checks, feature engineering, model construction, evaluation, and registry operations.

MLflow is the system of record for metrics, artifacts, dataset lineage, and champion alias management.

## Gates

The integrity gate blocks batches with missing required columns, invalid datetime coverage, impossible durations, severe target missingness, or broad numeric range violations. NannyML alerts are treated as soft warnings that influence retraining sensitivity.

The model gate compares champion RMSE on the current batch with champion RMSE on the reference batch. Retraining is recommended when degradation exceeds 3 percent with integrity warnings, or 5 percent for a clean batch.

The promotion gate compares candidate and champion performance on the same batch while checking reference stability. This prevents a candidate from being promoted just because it overfits the newest batch.

## Failure Recovery

Metaflow resume support allows failed downstream steps to restart without re-running completed data validation and feature engineering work. That matters when batch processing or candidate training becomes expensive.
