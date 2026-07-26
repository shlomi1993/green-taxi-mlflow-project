# Operations

## Local Services

Start MLflow with:

```bash
mlflow server --workers 1 --port 5000 \
  --backend-store-uri sqlite:///mlflow_tracking/mlflow.db \
  --default-artifact-root mlflow_tracking/mlruns
```

The UI is available at `http://localhost:5000`. The default backend store is SQLite and the artifact root is `mlflow_tracking/mlruns`.

## Runbook

1. Put incoming TLC parquet files in `data/raw/`.
2. Run the Metaflow workflow with the current reference and batch paths.
3. Review MLflow tags `decision_action`, `retrain_recommended`, and `promotion_recommended`.
4. Inspect `decision.json` for the exact gate rationale.
5. If promotion succeeds, downstream serving should reload `models:/green_taxi_tip_model@champion`.

## Serving

Serve the current champion locally:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m "models:/green_taxi_tip_model@champion" -p 5001 --env-manager local
```

After a promotion, restart the serving process to load the updated champion alias.
