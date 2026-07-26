import logging
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import os
import pandas as pd

from metaflow import FlowSpec, Parameter, step
from mlflow.tracking import MlflowClient

from green_taxi_tip_pipeline import (
    DEFAULT_EXPERIMENT,
    DEFAULT_URI,
    FEATURE_COLS,
    MIN_IMPROVEMENT_PCT,
    MODEL_NAME,
    DecisionAction,
    ModelRegistry,
    build_model,
    engineer_features,
    evaluate_model,
    load_taxi_table,
    log_decision,
    run_integrity_checks,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class GreenTaxiTipFlow(FlowSpec):
    """
    Batch monitoring loop:
        new batch -> integrity gate -> feature engineering -> performance gate -> retrain -> promote
    """
    batch_path = Parameter(
        "batch-path",
        help="Path to new batch parquet (e.g. 2020-04)",
        type=str,
        required=True,
    )
    ref_path = Parameter(
        "ref-path",
        help="Path to reference parquet (e.g. 2020-01)",
        type=str,
        default=None,
    )
    tracking_uri = Parameter(
        "tracking-uri",
        help="MLflow tracking URI",
        type=str,
        default=DEFAULT_URI,
    )
    experiment_name = Parameter(
        "experiment-name",
        help="MLflow experiment name",
        type=str,
        default=DEFAULT_EXPERIMENT,
    )
    model_name = Parameter(
        "model-name",
        help="MLflow registered model name",
        type=str,
        default=MODEL_NAME,
    )
    min_improvement = Parameter(
        "min-improvement",
        help="Minimum RMSE improvement fraction to promote (default 1%%)",
        type=float,
        default=MIN_IMPROVEMENT_PCT,
    )

    def init_mlflow(self) -> None:
        """
        Apply tracking configuration in each step process.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    # Pre-workflow - Initialize MLflow
    @step
    def start(self):
        """
        Initialize flow state, tracking, and model registry access for the run.
        """
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize attributes to be used across steps
        self.decision_action = None
        self.integrity_warn = False
        self.champion_model = None
        self.champion_uri = None
        self.candidate_model_uri = None
        self.candidate_rmse_batch = None
        self.candidate_rmse_ref = None

        # Initialize MLflow tracking and registry
        self.init_mlflow()
        self.registry = ModelRegistry(MlflowClient(), self.model_name)

        # Log configuration and start flow
        self.logger.info(f"MLflow configured: {self.tracking_uri} / experiment={self.experiment_name}")
        self.next(self.load_data)

    # Step A - Load Data
    @step
    def load_data(self):
        """
        Load reference and batch datasets from specified paths.
        If no reference path is provided, use the batch as the reference (baseline run).
        """
        # Load batch dataset
        self.df_batch = load_taxi_table(self.batch_path)

        # Load reference dataset, or use batch as reference if not provided
        if self.ref_path is None:
            self.logger.info("No reference path provided - using batch as reference (baseline run)")
            self.df_ref = self.df_batch.copy()
        else:
            self.df_ref = load_taxi_table(self.ref_path)

        # Log dataset sizes and proceed to integrity gate
        self.logger.info(f"Loaded {len(self.df_ref)} reference rows and {len(self.df_batch)} batch rows")
        self.next(self.integrity_gate)

    # Step B - Integrity Gate
    @step
    def integrity_gate(self):
        """
        Perform integrity checks on the reference and batch datasets.
        - Layer 1: hard rules (missing columns, invalid datetimes, negative durations, range violations) - "reject_batch" if any fail
        - Layer 2: NannyML soft checks (missingness drift, unseen categoricals) - sets integrity_warn tag
        """
        self.init_mlflow()

        with mlflow.start_run(run_name="integrity_gate") as run:
            self.integrity_run_id = run.info.run_id  # Capture run ID for testing
            mlflow.set_tag("pipeline_step", "integrity_gate")
            mlflow.set_tag("batch_path", self.batch_path)
            mlflow.set_tag("ref_path", self.ref_path)

            # Perform hard and soft integrity checks
            ok, report = run_integrity_checks(self.df_ref, self.df_batch)

            # Log metrics and artifacts
            mlflow.log_metrics(report.metrics)
            mlflow.log_dict({"hard_failures": report.hard.failures}, "hard_failures.json")
            mlflow.log_dict({"nannyml_details": report.soft.details}, "nannyml_details.json")

            # Make decision based on integrity results
            if ok:
                self.logger.info("Hard integrity checks passed - accepting batch")

                # Log NannyML warnings as MLflow tag and decision details
                nml_warn = report.soft.warn
                self.integrity_warn = nml_warn
                mlflow.set_tag("integrity_warn", str(nml_warn).lower())
                if nml_warn:
                    self.logger.warning("Soft integrity checks (NannyML) raised warnings")

                # Log acceptance decision
                self.decision_action = DecisionAction.BATCH_ACCEPTED
                log_decision(
                    action=self.decision_action,
                    reason="Hard rules passed" + (f"; {len(report.soft.details)} NannyML warnings present" if nml_warn else ""),
                    metrics=report.hard.metrics,
                    details={"nannyml_warn": nml_warn, "nannyml_details": report.soft.details},
                )

            else:
                self.logger.error("Hard integrity checks failed - rejecting batch")
                self.integrity_warn = False

                # Log rejection decision
                self.decision_action = DecisionAction.REJECT_BATCH
                log_decision(
                    action=self.decision_action,
                    reason="; ".join(report.hard.failures),
                    metrics=report.hard.metrics,

                )

        # Log results and proceed to feature engineering if batch accepted, otherwise end flow
        self.integrity_route = "accepted" if ok else "rejected"
        self.logger.info(f"Integrity gate completed: batch {self.integrity_route}" + (" with warnings" if self.integrity_warn else ""))
        self.next({"accepted": self.feature_engineering, "rejected": self.end}, condition="integrity_route")

    # Step C - Feature Engineering
    @step
    def feature_engineering(self):
        """
        Perform feature engineering on both reference and batch datasets, including:
        - Builds feature matrices (X) and target vectors (y)
        - Produces a stable feature schema defined by FEATURE_COLS (16 columns)
        - Applies domain transforms (credit-card filter, datetime/duration features, clipping, log transforms, etc.)
        - Logs feature schema and dtypes to MLflow as feature_cols.json
        """
        self.init_mlflow()

        # Log feature spec to MLflow
        with mlflow.start_run(run_name="feature_engineering") as run:
            self.feature_engineering_run_id = run.info.run_id  # Capture run ID for testing
            mlflow.set_tag("pipeline_step", "feature_engineering")

            # Perform feature engineering inside MLflow run context for tracking
            self.X_ref, self.y_ref = engineer_features(self.df_ref)
            self.X_batch, self.y_batch = engineer_features(self.df_batch)

            feature_spec = {col: str(self.X_ref[col].dtype) for col in FEATURE_COLS}
            mlflow.log_dict({"feature_cols": FEATURE_COLS, "dtypes": feature_spec}, "feature_cols.json")

        # Log results and proceed to load champion
        self.logger.info(f"Features engineered: {len(self.X_ref)} reference rows, {len(self.X_batch)} batch rows")
        self.next(self.load_champion)

    # Step D - Load Champion
    @step
    def load_champion(self):
        """
        Load the current champion model from the registry if exists, otherwise bootstrap a champion by training on the
        reference data and registering it as the champion with @champion alias.
        """
        self.init_mlflow()
        needs_bootstrap = not self.registry.champion_exists()

        # Guard against stale registry state where alias lookup fails at load time.
        if not needs_bootstrap:
            try:
                self.champion_model, self.champion_uri = self.registry.load_champion()
            except mlflow.exceptions.MlflowException as e:
                if "alias champion not found" in str(e).lower():
                    self.logger.warning("Champion alias missing at load time - bootstrapping a new champion")
                    needs_bootstrap = True
                else:
                    raise

        # Bootstrap champion if none exists (first run or alias missing)
        if needs_bootstrap:
            self.logger.info("No champion found - bootstrapping initial model on reference data")

            with mlflow.start_run(run_name="bootstrap_train") as run:
                self.bootstrap_train_run_id = run.info.run_id  # Capture run ID for testing
                mlflow.set_tag("pipeline_step", "train")
                mlflow.set_tag("bootstrap", "true")

                # Train model on reference data
                model = build_model()
                model.fit(self.X_ref, self.y_ref)

                # Evaluate and log metrics
                metrics = evaluate_model(model, self.X_ref, self.y_ref)
                mlflow.log_metrics({f"train_{k}": v for k, v in metrics.as_dict().items()})
                mlflow.log_dict({"feature_cols": FEATURE_COLS}, "feature_cols.json")

                # Log model and metadata to MLflow
                model_info = mlflow.sklearn.log_model(model, name="model", input_example=self.X_ref.head(5))
                mlflow.set_tag("model_uri", model_info.model_uri)

                # Register champion
                tags = {
                    "role": "champion",
                    "trained_on": "reference",
                    "validation_status": "approved",
                    "bootstrap": "true"
                }
                version = self.registry.register_version(model_info.model_uri, tags)
                self.registry.promote_to_champion(version, reason="bootstrap")
                self.logger.info(f"Registered bootstrap champion version {version}")

            # Load champion after bootstrap
            self.champion_model, self.champion_uri = self.registry.load_champion()

        # Load champion in normal path (when no bootstrap was needed)
        if self.champion_model is None:
            self.champion_model, self.champion_uri = self.registry.load_champion()
        self.logger.info(f"Champion loaded: {self.champion_uri}")
        self.next(self.model_gate)

    # Step E - Model Gate
    @step
    def model_gate(self):
        """
        Evaluate the champion model on the new batch and reference datasets, log results to MLflow, and decide whether
        retraining is needed based on performance changes and integrity warnings.
        If the champion's RMSE on the batch has increased by more than 3% (with integrity warnings) or 5% (without
        warnings) compared to the reference, then retraining is recommended.
        """
        self.init_mlflow()

        with mlflow.start_run(run_name="model_gate") as run:
            self.model_gate_run_id = run.info.run_id  # Capture run ID for testing
            mlflow.set_tag("pipeline_step", "model_gate")
            mlflow.set_tag("champion_uri", self.champion_uri)
            mlflow.set_tag("integrity_warn", str(self.integrity_warn).lower())

            # Evaluate champion on batch
            champ_metrics = evaluate_model(self.champion_model, self.X_batch, self.y_batch)
            mlflow.log_metrics({f"champion_{k}": v for k, v in champ_metrics.as_dict().items()})

            # Evaluate champion on reference (baseline)
            ref_metrics = evaluate_model(self.champion_model, self.X_ref, self.y_ref)
            mlflow.log_metrics({f"baseline_{k}": v for k, v in ref_metrics.as_dict().items()})

            # Log evaluation dataset lineage (P1 requirement)
            eval_dataset = mlflow.data.from_pandas(self.X_batch, name="batch_eval")
            mlflow.log_input(eval_dataset, context="evaluation")

            # Inference demo: log champion predictions on batch as artifact
            y_pred_champ = self.champion_model.predict(self.X_batch)
            pred_df = self.X_batch.copy()
            pred_df["prediction"] = y_pred_champ
            pred_df["actual"] = self.y_batch
            pred_path = "predictions.parquet"
            pred_df.to_parquet(pred_path, index=False)
            mlflow.log_artifact(pred_path)
            os.remove(pred_path)  # Clean up local file after logging

            # Calculate RMSE increase
            rmse_champion = champ_metrics.rmse
            rmse_baseline = ref_metrics.rmse
            rmse_increase_pct = (rmse_champion - rmse_baseline) / max(rmse_baseline, 1e-9) if rmse_baseline > 0 else 0.0
            mlflow.log_metric("rmse_increase_pct", rmse_increase_pct)

            # Store champion metrics for promotion gate
            self.rmse_champion_on_batch = rmse_champion
            self.rmse_champion_on_ref = rmse_baseline

            # Decision: retrain if RMSE increased beyond a threshold of 3% with integrity warnings or 5% otherwise
            threshold = 0.03 if self.integrity_warn else 0.05
            retrain_needed = rmse_increase_pct > threshold

            # Set retrain reason and action
            if retrain_needed:
                retrain_reason = f"RMSE increased by {rmse_increase_pct:.2%} which is above the threshold of {threshold:.2%}"
                self.decision_action = DecisionAction.RETRAIN
            else:
                retrain_reason = f"RMSE increase of {rmse_increase_pct:.2%} is within acceptable threshold of {threshold:.2%}"
                self.decision_action = DecisionAction.NO_RETRAIN

            # Prepare metrics
            metrics = {
                "rmse_champion_on_batch": rmse_champion,
                "rmse_champion_on_ref": rmse_baseline,
                "rmse_increase_pct": rmse_increase_pct
            }

            # Log decision with details
            log_decision(action=self.decision_action, retrain_recommended=retrain_needed, reason=retrain_reason, metrics=metrics)

        # Log results and proceed to retrain if needed, or end flow otherwise
        self.logger.info(f"Model gate completed: retrain is {'needed' if retrain_needed else 'not needed'} because {retrain_reason}")
        self.model_gate_route = "retrain" if retrain_needed else "end"
        self.next({"retrain": self.retrain, "end": self.end}, condition="model_gate_route")

    # Step F - Retrain
    @step
    def retrain(self):
        """
        If needed, retrain the model on the combined reference and batch datasets, log results to MLflow, and prepare
        outputs for the promotion gate.
        The retrained candidate model is evaluated on both the batch (for performance) and reference (for stability)
        datasets, and metrics are logged to MLflow.
        """
        # raise RuntimeError("Simulated failure for demo")  # NOTE Inject error for demo
        self.init_mlflow()

        with mlflow.start_run(run_name="retrain") as run:
            self.retrain_run_id = run.info.run_id  # Capture run ID for testing
            mlflow.set_tag("pipeline_step", "retrain")
            mlflow.set_tag("trained_on_batches", f"{self.ref_path},{self.batch_path}")

            # Train on merged reference and batch data
            X_train = pd.concat([self.X_ref, self.X_batch], ignore_index=True)
            y_train = np.concatenate([self.y_ref, self.y_batch])

            # Train model
            model = build_model()
            model.fit(X_train, y_train)

            # Log training dataset lineage
            train_dataset = mlflow.data.from_pandas(X_train, name="train_combined")
            mlflow.log_input(train_dataset, context="training")

            # Evaluate candidate on batch (same eval set as champion)
            candidate_metrics = evaluate_model(model, self.X_batch, self.y_batch)
            mlflow.log_metrics({f"candidate_{k}": v for k, v in candidate_metrics.as_dict().items()})

            # Evaluate candidate on reference (stability check - P3)
            ref_metrics = evaluate_model(model, self.X_ref, self.y_ref)
            mlflow.log_metrics({f"candidate_ref_{k}": v for k, v in ref_metrics.as_dict().items()})

            # Log model and metadata to MLflow
            model_info = mlflow.sklearn.log_model(model, name="model", input_example=self.X_batch.head(5))
            mlflow.set_tag("model_uri", model_info.model_uri)
            mlflow.log_dict({"feature_cols": FEATURE_COLS}, "feature_cols.json")
            mlflow.log_params({"ref_path": self.ref_path, "batch_path": self.batch_path})

            # Inference demo: log batch predictions as artifact
            y_pred = model.predict(self.X_batch)
            pred_df = self.X_batch.copy()
            pred_df["prediction"] = y_pred
            pred_df["actual"] = self.y_batch
            pred_path = "predictions.parquet"
            pred_df.to_parquet(pred_path, index=False)
            mlflow.log_artifact(pred_path)
            os.remove(pred_path)

        # Set retrain outputs for promotion gate
        self.candidate_model_uri = model_info.model_uri
        self.candidate_rmse_batch = candidate_metrics.rmse
        self.candidate_rmse_ref = ref_metrics.rmse

        # Log results and proceed to promotion gate
        self.logger.info(f"Retrain done: candidate RMSE on batch={self.candidate_rmse_batch:.4f}, on reference={self.candidate_rmse_ref:.4f}")
        self.next(self.promotion_gate)

    # Step G - Promotion Gate
    @step
    def promotion_gate(self):
        """
        Evaluate the retrained candidate model against the champion using the following criteria:
        - P1: Evaluation is valid - champion and candidate metrics exist, evaluation dataset logged
        - P2: Candidate beats champion meaningfully - e.g. >1% improvement in RMSE on batch
        - P3: Stability - candidate doesn't regress on reference by >5%
        - P4: No hard integrity failures - guaranteed by flow structure
        If all criteria are met, the candidate is promoted to champion in the registry.
        Otherwise, it is rejected but still registered with a "rejected" tag for audit trail.
        """
        self.init_mlflow()

        with mlflow.start_run(run_name="promotion_gate") as run:
            self.promotion_gate_run_id = run.info.run_id  # Capture run ID for testing
            mlflow.set_tag("pipeline_step", "promotion_gate")

            # Collect metrics for decision making and log them
            rmse_champ = self.rmse_champion_on_batch
            rmse_cand = self.candidate_rmse_batch
            rmse_cand_ref = self.candidate_rmse_ref

            # P1: Evaluation is valid
            p1 = True  # This gate is only reachable if evaluation metrics exist and evaluation dataset is logged

            # P2: Candidate beats champion meaningfully
            threshold = rmse_champ * (1 - self.min_improvement)
            p2 = rmse_cand < threshold

            # P3: Stability - candidate doesn't regress on reference by >5%
            ref_regression = (rmse_cand_ref - self.rmse_champion_on_ref) / max(self.rmse_champion_on_ref, 1e-9)
            tolerance = 0.05
            p3 = ref_regression < tolerance

            # P4: No hard integrity failures
            p4 = True  # This gate is only reachable if batch passed integrity checks

            # Make decision (p1 and p4 are always true, so decision is p2 and p3)
            promote = p2 and p3

            # Build reason string
            reasons = []
            if promote:
                reasons.append("evaluation metrics valid and dataset logged")
            if not p2:
                reasons.append(f"candidate RMSE {rmse_cand:.4f} >= {threshold:.4f}")
            else:
                reasons.append(f"candidate RMSE {rmse_cand:.4f} < {threshold:.4f} "
                               f"(>{self.min_improvement:.0%} better than champion RMSE {rmse_champ:.4f})")
            if not p3:
                reasons.append(f"reference regression {ref_regression:.2%} > {tolerance:.0%}")
            else:
                reasons.append(f"reference regression {ref_regression:.2%} <= {tolerance:.0%}")
            if promote:
                reasons.append("no hard integrity failures")
            reason = "; ".join(reasons)

            # Prepare metrics
            metrics = {
                "champion_rmse_batch": rmse_champ,
                "candidate_rmse_batch": rmse_cand,
                "candidate_rmse_ref": rmse_cand_ref,
                "champion_rmse_ref": self.rmse_champion_on_ref,
                "min_improvement_pct": self.min_improvement,
                "ref_regression_pct": ref_regression
            }

            # Log decision
            self.decision_action = DecisionAction.PROMOTE if promote else DecisionAction.NO_PROMOTE
            log_decision(
                action=self.decision_action,
                promotion_recommended=promote,
                reason=reason,
                metrics=metrics,
                details={"p1": p1, "p2": p2, "p3": p3, "p4": p4},
            )

            # Register candidate if promoted and promote to champion
            if promote:
                tags = {
                    "role": "candidate",
                    "trained_on_batches": f"{self.ref_path},{self.batch_path}",
                    "eval_batch_id": self.batch_path,
                    "validation_status": "approved",
                    "decision_reason": reason,
                }
                version = self.registry.register_version(self.candidate_model_uri, tags)
                self.registry.promote_to_champion(version, reason=reason)
                mlflow.set_tag("promoted_version", version)
                self.logger.info(f"PROMOTED candidate to champion version {version}")

            # Otherwise log rejection and register as rejected candidate for audit trail
            else:
                self.logger.info(f"Promotion declined: {reason}")
                if self.candidate_model_uri:
                    tags = {
                        "role": "candidate",
                        "trained_on_batches": f"{self.ref_path},{self.batch_path}",
                        "eval_batch_id": self.batch_path,
                        "validation_status": "rejected",
                        "decision_reason": reason,
                    }
                    version = self.registry.register_version(self.candidate_model_uri, tags)

        # Log promotion results and end flow
        self.logger.info(f"Promotion gate completed: mode {'PROMOTED' if promote else 'REJECTED'} because {reason}")
        self.next(self.end)

    # End
    @step
    def end(self):
        """
        End the workflow and log the final decision outcome.
        """
        outcome_messages = {
            DecisionAction.REJECT_BATCH: "Batch was rejected by integrity gate",
            DecisionAction.NO_RETRAIN: "Champion is adequate, no retrain needed",
            DecisionAction.NO_PROMOTE: "Retrained but candidate was NOT promoted",
            DecisionAction.PROMOTE: "Candidate was PROMOTED to champion",
        }
        assert self.decision_action in outcome_messages.keys(), f"Unknown decision action: {self.decision_action}"
        self.logger.info(f"Flow finished: {outcome_messages[self.decision_action]}")
