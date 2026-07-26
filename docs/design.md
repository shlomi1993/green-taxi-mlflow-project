# Green Taxi Tip Model Monitoring Platform

## Engineering Design

**Status:** Implemented  
**Owners:** ML Platform / Model Operations  
**Last updated:** July 2026

## 1. Executive Summary

This system operates a batch-trained regression model that predicts tip amounts for NYC Green Taxi trips. It validates newly arrived labeled data, measures the production champion's performance, retrains when degradation crosses an explicit threshold, and promotes a candidate only when it improves current-batch performance without materially regressing on the stable reference population.

Metaflow provides durable workflow boundaries and restartability. MLflow is the system of record for experiments, artifacts, dataset lineage, model versions, and the `@champion` alias. NannyML provides statistical drift signals used to adjust the sensitivity of the performance gate.

The current implementation targets a single-node batch operating model. It is suitable for scheduled local or VM execution and is intentionally structured so storage, orchestration, and serving can later move to managed infrastructure without rewriting the core model logic.

## 2. Goals and Non-Goals

### Goals

- Reject malformed or operationally unsafe batches before model evaluation.
- Detect material champion degradation on newly labeled data.
- Produce reproducible features for training, evaluation, and inference.
- Retrain automatically when the configured performance policy requires it.
- Prevent promotion of candidates that overfit the latest batch.
- Preserve an auditable record of every gate and registry transition.
- Resume failed workflows from persisted Metaflow step boundaries.
- Keep operational logic independent from notebooks and ad hoc execution environments.

### Non-Goals

- Online or low-latency inference orchestration.
- Real-time feature computation or streaming drift detection.
- Automated hyperparameter optimization.
- Multi-model traffic splitting, shadow deployments, or canary rollout.
- Automatic rollback based on live serving telemetry.
- Cloud-specific scheduling, secrets management, or distributed execution.

## 3. System Context

The workflow consumes labeled TLC trip records in Parquet or CSV format. A reference dataset represents the stable comparison population and a batch dataset represents the new operating period.

```text
TLC batch storage
      |
      v
Metaflow monitoring workflow
      |
      +--> integrity and drift checks
      +--> deterministic feature engineering
      +--> champion evaluation
      +--> conditional candidate training
      +--> promotion policy
      |
      v
MLflow tracking, artifacts, and model registry
      |
      v
models:/green_taxi_tip_model@champion
```

The implementation is divided into two primary modules:

- `green_taxi_flow.py` owns orchestration, branching, and MLflow run boundaries.
- `src/pipeline.py` owns data loading, validation, feature engineering, model construction, evaluation, decision logging, and registry operations.

## 4. Data Contract

### Required input columns

| Column | Purpose |
| --- | --- |
| `lpep_pickup_datetime` | Temporal features and duration validation |
| `lpep_dropoff_datetime` | Trip duration |
| `trip_distance` | Distance and interaction features |
| `fare_amount` | Fare and interaction features |
| `tip_amount` | Training and evaluation target |
| `PULocationID` | Pickup location features |
| `DOLocationID` | Drop-off location features |
| `passenger_count` | Numeric model feature |
| `payment_type` | Credit-card population filter and drift check |

The target is required because the current workflow performs delayed, labeled batch monitoring. Unlabeled production inference is outside this workflow.

### Hard validation policy

A batch is rejected when it has missing required columns, predominantly invalid timestamps, more than 50 percent missing targets, negative trip durations, or excessive violations of configured numeric ranges. A rejected batch cannot reach feature engineering, retraining, or model registration.

### Soft validation policy

NannyML checks missing-value drift, univariate distribution drift, and multivariate reconstruction drift. The workflow also measures unseen values in selected categorical columns. These findings do not reject the batch; they set `integrity_warn=true` and make the retraining gate more sensitive.

## 5. Feature and Model Design

Feature engineering produces a stable 16-column schema:

- Raw numeric values: distance, fare, passenger count, and trip duration.
- Log transforms for distance, fare, and duration.
- Pickup hour, weekday, and month.
- Pickup and drop-off zone identifiers.
- Pickup and drop-off frequency encodings.
- Distance-per-minute and fare-per-mile interactions.

The implementation filters to credit-card trips, clips operational outliers, and uses a median imputer inside the scikit-learn pipeline. The estimator is a `GradientBoostingRegressor`. The same feature function and schema are used for bootstrap training, champion evaluation, candidate training, and prediction artifacts.

The feature list and dtypes are logged to MLflow so schema changes are visible and reviewable.

## 6. Workflow and Decision Policy

### 6.1 Load data

The flow loads the incoming batch and an optional reference dataset. When no reference path is supplied, the batch is copied as the reference; this is the bootstrap/baseline operating mode.

### 6.2 Integrity gate

Hard checks run first. Failure produces `decision.json` with `action=reject_batch` and terminates the run. A passing batch proceeds with any soft warnings recorded as MLflow metrics, tags, and NannyML detail artifacts.

### 6.3 Feature engineering

Reference and batch data are transformed independently through the same deterministic feature function. The resulting schema is logged before any model operation.

### 6.4 Champion resolution

The registry is queried for:

```text
models:/green_taxi_tip_model@champion
```

If the alias does not exist, the workflow trains on the reference population, registers version 1, marks it approved, and assigns the `@champion` alias. A stale or missing alias is handled as a bootstrap condition rather than silently selecting an arbitrary model version.

### 6.5 Performance gate

The champion is evaluated on both the current batch and the reference population.

```text
rmse_increase =
    (batch_rmse - reference_rmse) / max(reference_rmse, epsilon)
```

Retraining is triggered when:

- `rmse_increase > 3%` for a batch with integrity warnings; or
- `rmse_increase > 5%` for a clean batch.

The gate logs RMSE, MAE, R², dataset lineage, batch predictions, and the complete decision rationale.

### 6.6 Candidate training

When retraining is required, reference and batch feature sets are combined into an expanding training window. The candidate uses the same estimator pipeline and feature contract as the champion. Candidate performance is measured on both the current batch and the reference population.

### 6.7 Promotion gate

Promotion requires every condition below:

1. Candidate evaluation and lineage exist.
2. Candidate batch RMSE is more than 1 percent better than champion batch RMSE.
3. Candidate reference RMSE regresses by less than 5 percent.
4. The batch passed all hard integrity checks.

An accepted candidate is registered and assigned the `@champion` alias. The previous version is tagged as the prior champion. A rejected candidate is still registered with `validation_status=rejected`, preserving the audit trail without exposing it to consumers.

The alias update is the deployment contract. Serving systems resolve the champion alias and must reload it after promotion.

## 7. Observability and Auditability

Each control point creates an MLflow run with a `pipeline_step` tag. Operational evidence includes:

- Input paths and dataset lineage.
- Integrity metrics, hard failures, and NannyML details.
- Feature names and dtypes.
- Champion and candidate evaluation metrics.
- Prediction Parquet artifacts with actual and predicted values.
- `retrain_recommended`, `promotion_recommended`, and integrity tags.
- `decision.json` containing policy inputs, outcomes, and rationale.
- Registered-model version tags and champion alias transitions.

Metaflow records step state independently from MLflow. This separation is intentional: Metaflow is responsible for execution recovery, while MLflow is responsible for model and decision history across workflow runs.

## 8. Failure Handling

| Failure | Behavior |
| --- | --- |
| Missing or unreadable input | Flow fails before model mutation |
| Hard data-quality violation | Batch is rejected and decision is logged |
| NannyML calculation failure | Recorded as soft-check detail; hard checks still govern acceptance |
| Feature or training failure | Metaflow marks the step failed; no downstream promotion occurs |
| Missing champion alias | Workflow bootstraps a new champion |
| Candidate fails policy | Version is registered as rejected; alias remains unchanged |
| Registry or artifact-store outage | Step fails and can be resumed after service recovery |

Completed Metaflow steps are checkpointed. Operators can correct a transient issue and use Metaflow `resume` without re-running successful upstream work.

## 9. Security and Governance

- Tracking URIs and service configuration should be supplied through environment or runtime parameters, not committed credentials.
- Production MLflow should use authenticated transport and a durable database and artifact store; the repository's SQLite configuration is for local operation.
- Write access to model aliases should be restricted to the workflow identity.
- Raw trip records and prediction artifacts may contain sensitive operational data and should follow the organization's retention and access-control policies.
- Model promotion is policy-driven and fully logged, but the current implementation does not include human approval or cryptographic artifact signing.

## 10. Deployment and Operations

The local reference deployment consists of one MLflow server backed by SQLite and local artifacts plus a locally executed Metaflow process. For a production deployment:

- Schedule the flow using the organization's orchestrator or Metaflow backend.
- Replace SQLite with a managed relational database.
- Store artifacts in versioned object storage.
- Run under a dedicated service identity with least-privilege registry access.
- Emit workflow and MLflow service metrics to centralized monitoring.
- Configure alerts for rejected batches, failed runs, repeated retraining, and registry update failures.
- Pin dependencies with a generated lock file and build an immutable execution image.

Serving consumers load `models:/green_taxi_tip_model@champion`. Alias changes should trigger a controlled model reload rather than relying on an indefinitely cached model.

## 11. Reliability Targets

The repository does not enforce external service-level objectives, but a production deployment should define at least:

- Maximum acceptable time from labeled batch arrival to a completed decision.
- Maximum consecutive failed or rejected batches.
- MLflow tracking and registry availability.
- Artifact-retention period and recovery objectives.
- Alert response time for failed promotion or champion-loading events.

These targets depend on batch cadence and downstream business requirements and should be set by the operating team.

## 12. Known Limitations and Follow-Up Work

- Thresholds are configured in code and documented in YAML; loading all quality-gate values directly from a validated runtime configuration is future work.
- Frequency encoding is fitted independently per dataset. A persisted training-time encoder would provide stricter train/serve consistency.
- Candidate training uses an expanding window rather than an explicit retention or weighting policy.
- NannyML failures are soft by design; production policy may need severity-based blocking for critical monitors.
- Promotion and alias update are not wrapped in a cross-system transaction.
- No automatic serving rollout, rollback, shadow test, or post-deployment health gate is implemented.
- The test suite exercises gate behavior extensively, but production deployment should add integration tests against the chosen remote MLflow backend and artifact store.

## 13. Design Decisions

| Decision | Rationale | Trade-off |
| --- | --- | --- |
| Labeled batch monitoring | Enables direct regression-performance measurement | Decisions wait for labels |
| Hard and soft integrity layers | Separates unsafe data from statistical change | Soft alerts require policy tuning |
| RMSE-relative retraining thresholds | Easy to interpret and audit | Sensitive to a very small reference RMSE |
| Expanding training window | Simple and retains historical coverage | Old regimes may receive too much weight |
| Reference stability promotion check | Reduces one-batch overfitting | Can reject useful adaptation to structural change |
| MLflow champion alias | Stable consumer contract and auditable versions | Consumers must implement reload behavior |
| Metaflow step persistence | Supports recovery without repeating expensive work | Local backend is not highly available |
