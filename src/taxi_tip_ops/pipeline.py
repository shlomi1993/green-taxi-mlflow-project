
import mlflow
import nannyml as nml
import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass, field
from enum import Enum
from mlflow.tracking import MlflowClient
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from typing import Any, Dict, List, Optional, Tuple, Union

RAW_DATETIME_COLS = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime"
]

REQUIRED_COLS = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "payment_type",
]

RANGE_RULES = [
    ("trip_distance", 0.0, 200.0),
    ("fare_amount", 0.0, 500.0),
    ("tip_amount", 0.0, 200.0),
    ("passenger_count", 0.0, 10.0),
]

# Raw numeric columns that exist in data before feature engineering for NannyML drift detection on raw batches
RAW_NUMERIC_COLS = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "passenger_count",
]

MIN_IMPROVEMENT_PCT = 0.01  # 1 % default

FEATURE_COLS = [

    # Original numeric features
    "trip_distance",
    "fare_amount",
    "passenger_count",
    "duration_min",

    # Log-transformed heavy-tailed features
    "log_trip_distance",
    "log_fare_amount",
    "log_duration_min",

    # Temporal features
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",

    # Location features (raw IDs)
    "PULocationID",
    "DOLocationID",

    # Location frequency encoding
    "PU_frequency",
    "DO_frequency",

    # Interaction features
    "distance_per_minute",
    "fare_per_mile",
]

TARGET_COL = "tip_amount"
MODEL_NAME = "green_taxi_tip_model"
DEFAULT_URI = "http://localhost:5001"
DEFAULT_EXPERIMENT = "taxi_tip_monitoring"


def load_taxi_table(path: Union[str, Path]) -> pd.DataFrame:
    """
    Data Loading: Load a taxi dataset from the given path (Parquet or CSV), ensuring datetime columns are parsed.

    Args:
        path (str or Path): Path to the dataset file (Parquet or CSV).

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    for col in RAW_DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


@dataclass
class HardIntegrityResult:
    """
    Result of hard integrity checks.
    """
    passed: bool
    failures: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def run_hard_integrity_checks(df: pd.DataFrame) -> HardIntegrityResult:
    """
    Hard Integrity Checks: Hard rules that cause immediate batch rejection (fail-fast).

    Checks:
        1. Required columns present - all columns in REQUIRED_COLS must be present.
        2. Target column exists and has values - 'tip_amount' must exist and not be mostly null.
        3. Datetime validity - 'lpep_pickup_datetime' and 'lpep_dropoff_datetime' must parse to valid datetimes.
        4. Negative duration - 'lpep_dropoff_datetime' must not be before 'lpep_pickup_datetime'.
        5. Range violations - numeric columns must fall within specified ranges.

    Args:
        df (pd.DataFrame): Dataset to check, expected to have raw columns (e.g. lpep_pickup_datetime) for checks.

    Returns:
        HardIntegrityResult: Result of the checks, including the batch status, any failure messages, and metrics.
    """
    failures = []
    metrics = {}

    # 1. Required columns present
    present = set(df.columns)
    missing = [col for col in REQUIRED_COLS if col not in present]
    metrics["missing_required_cols"] = float(len(missing))
    if missing:
        failures.append(f"Missing required columns: {missing}")

    # 2. Target column exists and has values
    if TARGET_COL not in df.columns:
        failures.append("Target column 'tip_amount' missing entirely.")
    else:
        target_null_frac = float(df[TARGET_COL].isna().mean())
        metrics["target_null_frac"] = target_null_frac
        if target_null_frac > 0.5:
            failures.append(f"tip_amount null fraction too high: {target_null_frac:.2%}")

    # 3. Datetime validity
    for col in RAW_DATETIME_COLS:
        if col not in df.columns:
            continue
        nat_frac = float(pd.to_datetime(df[col], errors="coerce").isna().mean())
        metrics[f"{col}_nat_frac"] = nat_frac
        if nat_frac > 0.5:
            failures.append(f"{col} has {nat_frac:.2%} invalid datetimes")

    # 4. Negative duration (dropoff before pickup)
    if all(col in df.columns for col in RAW_DATETIME_COLS):
        dur = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds()
        neg_frac = float((dur < 0).mean()) if len(dur) else 0.0
        metrics["negative_duration_frac"] = neg_frac
        if neg_frac > 0.10:
            failures.append(f"Negative duration fraction: {neg_frac:.2%}")

    # 5. Range violations
    for col, lo, hi in RANGE_RULES:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x) == 0:
            continue
        bad = ((x < lo) | (x > hi)).mean()  # fraction of out-of-range values
        metrics[f"range_bad_{col}"] = float(bad)
        if bad > 0.20:
            failures.append(f"{col} out-of-range fraction: {bad:.2%}")

    return HardIntegrityResult(passed=len(failures) == 0, failures=failures, metrics=metrics)


@dataclass
class SoftIntegrityResult:
    """
    Result of NannyML soft integrity checks.
    """
    warn: bool = False
    details: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def run_soft_integrity_checks(df_ref: pd.DataFrame, df_cur: pd.DataFrame) -> SoftIntegrityResult:
    """
    Soft Integrity Checks: Run NannyML-based soft data quality checks.

    Checks:
        1. Missing value drift - using NannyML MissingValuesCalculator.
        2. Univariate distribution drift - using NannyML UnivariateDriftCalculator.
        3. Multivariate drift via reconstruction error - using NannyML DataReconstructionDriftCalculator.
        4. Unseen categorical values in key location features - comparing current batch to reference.

    These checks do not cause hard batch rejection but can trigger warnings and inform retraining decisions.

    Args:
        df_ref (pd.DataFrame): Reference dataset for comparison, typically historical data used for model training.
        df_cur (pd.DataFrame): Current batch dataset to check, expected to have the same schema as df_ref.

    Returns:
        SoftIntegrityResult: Result containing any warnings, details, and metrics.
    """
    # Identify numeric columns before feature engineering to check for drift
    num_cols = [col for col in RAW_NUMERIC_COLS if col in df_ref.columns and col in df_cur.columns]
    if not num_cols:
        return SoftIntegrityResult()  # No numeric features to check, return clean result

    warn = False
    details = []
    metrics = {}

    # Prepare reference and analysis DataFrames with synthetic timestamp
    ref_chunk = df_ref[num_cols].copy().reset_index(drop=True)
    ref_chunk["_nml_idx"] = range(len(ref_chunk))
    cur_chunk = df_cur[num_cols].copy().reset_index(drop=True)
    cur_chunk["_nml_idx"] = range(len(ref_chunk), len(ref_chunk) + len(cur_chunk))

    # 1. Missingness drift via NannyML
    try:
        missing_calc = nml.MissingValuesCalculator(
            column_names=num_cols,
            timestamp_column_name="_nml_idx",
            chunk_size=len(cur_chunk) or 1000,
        )
        missing_calc.fit(ref_chunk)
        missing_results = missing_calc.calculate(cur_chunk)

        for col_name in num_cols:
            col_results = missing_results.filter(column_names=[col_name])
            alerts = col_results.to_df()

            if "alert" in alerts.columns:
                n_alerts = int(alerts["alert"].sum())
            else:
                alert_cols = [c for c in alerts.columns if "alert" in str(c).lower()]
                n_alerts = int(alerts[alert_cols].sum().sum()) if alert_cols else 0

            metrics[f"nml_missing_alerts_{col_name}"] = float(n_alerts)
            if n_alerts > 0:
                warn = True
                details.append(f"NannyML missing value alert for '{col_name}'")
    except Exception as exc:
        details.append(f"NannyML missing value check failed: {exc}")

    # 2. Univariate drift via NannyML
    try:
        calculator = nml.UnivariateDriftCalculator(
            column_names=num_cols,
            timestamp_column_name="_nml_idx",
            chunk_size=len(cur_chunk) or 1000,
        )
        calculator.fit(ref_chunk)
        drift_results = calculator.calculate(cur_chunk)

        for col_name in num_cols:
            col_results = drift_results.filter(column_names=[col_name])
            alerts = col_results.to_df()
            if "alert" in alerts.columns:
                n_alerts = int(alerts["alert"].sum())
            else:  # Try column-specific alert columns
                alert_cols = [c for c in alerts.columns if "alert" in str(c).lower()]
                n_alerts = int(alerts[alert_cols].sum().sum()) if alert_cols else 0

            metrics[f"nml_drift_alerts_{col_name}"] = float(n_alerts)
            if n_alerts > 0:
                warn = True
                details.append(f"NannyML drift alert for '{col_name}'")

    except Exception as exc:
        details.append(f"NannyML univariate drift check failed: {exc}")

    # 3. Multivariate drift via NannyML Data Reconstruction (PCA-based)
    try:
        reconstruction_calc = nml.DataReconstructionDriftCalculator(
            column_names=num_cols,
            timestamp_column_name="_nml_idx",
            chunk_size=len(cur_chunk) or 1000,
        )
        reconstruction_calc.fit(ref_chunk)
        reconstruction_results = reconstruction_calc.calculate(cur_chunk)

        recon_df = reconstruction_results.to_df()  # Reconstruction alerts indicate multivariate drift
        if "alert" in recon_df.columns:
            n_alerts = int(recon_df["alert"].sum())
        else:  # Try column-specific alert columns
            alert_cols = [c for c in recon_df.columns if "alert" in str(c).lower()]
            n_alerts = int(recon_df[alert_cols].sum().sum()) if alert_cols else 0

        metrics["nml_multivariate_drift_alerts"] = float(n_alerts)
        if n_alerts > 0:
            warn = True
            details.append("NannyML multivariate drift detected (reconstruction error alert)")
    except Exception as exc:
        details.append(f"NannyML multivariate drift check failed: {exc}")

    # 4. Unseen categoricals
    cat_cols = ["PULocationID", "DOLocationID", "payment_type"]
    for col in cat_cols:
        if col not in df_ref.columns or col not in df_cur.columns:
            continue
        ref_vals = set(df_ref[col].dropna().unique())
        cur_vals = set(df_cur[col].dropna().unique())
        unseen = cur_vals - ref_vals
        frac = len(unseen) / max(len(cur_vals), 1)
        metrics[f"unseen_cats_{col}"] = float(frac)
        if frac > 0.10:
            warn = True
            details.append(f"Unseen categories in '{col}': {len(unseen)} new values")

    return SoftIntegrityResult(warn=warn, details=details, metrics=metrics)


@dataclass
class CombinedIntegrityReport:
    """
    Combined integrity report containing results from both hard and soft checks.
    """
    hard: HardIntegrityResult
    soft: SoftIntegrityResult = field(default_factory=SoftIntegrityResult)

    @property
    def metrics(self) -> Dict[str, float]:
        """
        Combined metrics from both hard and soft checks.

        Returns:
            Dict[str, float]: Merged metrics dictionary.
        """
        combined = dict(self.hard.metrics)  # Copy hard metrics
        if self.soft:
            combined.update(self.soft.metrics)  # Add/overwrite with soft metrics if available
        return combined


def run_integrity_checks(df_ref: pd.DataFrame, df_batch: pd.DataFrame) -> Tuple[bool, CombinedIntegrityReport]:
    """
    Combined hard and NannyML soft integrity checks.

    Args:
        df_ref (pd.DataFrame): Reference dataset for comparison (e.g. historical baseline).
        df_batch (pd.DataFrame): Current batch dataset to check.

    Returns:
        Tuple[bool, CombinedIntegrityReport]: Tuple of (passed, report) where passed indicates if hard checks passed.
    """
    # Layer 1: Hard integrity checks (fail-fast)
    hard = run_hard_integrity_checks(df_batch)
    report = CombinedIntegrityReport(hard=hard)
    if not hard.passed:
        return False, report

    # Layer 2: NannyML soft integrity checks (warnings, no hard fail)
    report.soft = run_soft_integrity_checks(df_ref, df_batch)
    return True, report


class DecisionAction(str, Enum):
    """
    Enum for decision action types.
    """
    REJECT_BATCH = "reject_batch"
    BATCH_ACCEPTED = "batch_accepted"
    NO_RETRAIN = "no_retrain"
    RETRAIN = "retrain"
    NO_PROMOTE = "no_promote"
    PROMOTE = "promote"


def log_decision(action: DecisionAction, retrain_recommended: bool = False, promotion_recommended: bool = False,
                 reason: str = "", metrics: Dict[str, float] = None, details: Dict[str, Any] = None) -> None:
    """
    Log a decision to MLflow.

    Args:
        action (DecisionAction): The action taken.
        retrain_recommended (bool, optional): Whether a retrain is recommended. Defaults to False.
        promotion_recommended (bool, optional): Whether a promotion is recommended. Defaults to False.
        reason (str, optional): Reason for the decision. Defaults to "".
        metrics (Dict[str, float], optional): Additional metrics. Defaults to None.
        details (Dict[str, Any], optional): Additional details. Defaults to None.
    """
    decision_dict = {
        "action": action.value,
        "retrain_recommended": retrain_recommended,
        "promotion_recommended": promotion_recommended,
        "reason": reason,
        "metrics": metrics or {},
        "details": details or {}
    }
    mlflow.log_dict(decision_dict, "decision.json")
    mlflow.set_tag("retrain_recommended", str(retrain_recommended).lower())
    mlflow.set_tag("promotion_recommended", str(promotion_recommended).lower())
    mlflow.set_tag("decision_action", action.value)


def engineer_features(df_raw: pd.DataFrame, credit_card_only: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Feature Engineering: Produce a model-ready (X, y) pair with a stable column schema.

    Steps:
      1. Filter to credit-card transactions (payment_type == 1) if requested.
      2. Derive calendar features from pickup datetime.
      3. Derive duration_min from pickup/dropoff.
      4. Clip heavy-tailed numerics.
      5. Log-transform heavy-tailed features.
      6. Encode location features with frequency encoding.
      7. Create interaction features.
      8. Select FEATURE_COLS and return (X, y).

    Args:
        df_raw (pd.DataFrame): Raw dataset loaded from source.
        credit_card_only (bool, optional): Whether to filter to credit-card transactions only. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Model-ready (X, y) pair.
    """
    df = df_raw.copy()

    # Filter
    if credit_card_only and "payment_type" in df.columns:
        df = df[df["payment_type"] == 1].copy()

    # Datetime features
    if "lpep_pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")
        df["pickup_hour"] = dt.dt.hour.astype("float64")
        df["pickup_weekday"] = dt.dt.dayofweek.astype("float64")
        df["pickup_month"] = dt.dt.month.astype("float64")

    # Duration
    if all(col in df.columns for col in RAW_DATETIME_COLS):
        dur = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60.0
        df["duration_min"] = dur
    elif "duration_min" not in df.columns:
        df["duration_min"] = np.nan

    # Clip heavy tails before log transform
    if "trip_distance" in df.columns:
        df["trip_distance"] = df["trip_distance"].clip(0, 100)
    if "fare_amount" in df.columns:
        df["fare_amount"] = df["fare_amount"].clip(0, 300)

    # Log transforms for heavy-tailed numeric fields
    if "trip_distance" in df.columns:
        df["log_trip_distance"] = np.log1p(df["trip_distance"].fillna(0))
    else:
        df["log_trip_distance"] = np.nan

    if "fare_amount" in df.columns:
        df["log_fare_amount"] = np.log1p(df["fare_amount"].fillna(0))
    else:
        df["log_fare_amount"] = np.nan

    if "duration_min" in df.columns:
        df["log_duration_min"] = np.log1p(df["duration_min"].clip(0, 1000).fillna(0))
    else:
        df["log_duration_min"] = np.nan

    # Location frequency encoding (stateless, computed per dataset)
    for loc_col, freq_col in [("PULocationID", "PU_frequency"), ("DOLocationID", "DO_frequency")]:
        if loc_col in df.columns:
            freq_map = df[loc_col].value_counts(normalize=True, dropna=False).to_dict()
            df[freq_col] = df[loc_col].map(freq_map).fillna(0.0)
        else:
            df[freq_col] = 0.0

    # Interaction features
    if "trip_distance" in df.columns and "duration_min" in df.columns:
        # Average speed proxy (distance per minute)
        df["distance_per_minute"] = df["trip_distance"] / df["duration_min"].clip(lower=1.0)
        df["distance_per_minute"] = df["distance_per_minute"].fillna(0).clip(0, 5)  # Cap outliers
    else:
        df["distance_per_minute"] = 0.0

    if "fare_amount" in df.columns and "trip_distance" in df.columns:
        # Price efficiency (fare per mile)
        df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"].clip(lower=0.1)
        df["fare_per_mile"] = df["fare_per_mile"].fillna(0).clip(0, 100)  # Cap outliers
    else:
        df["fare_per_mile"] = 0.0

    # Target
    y_raw: pd.Series = pd.to_numeric(df.get(TARGET_COL), errors="coerce")
    y = y_raw.fillna(0.0).to_numpy(dtype=float)

    # Select stable feature set, fill missing cols with NaN
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    X = df[FEATURE_COLS].astype("float64")

    return X, y


def build_model(random_state: int = 0, n_estimators: int = 200, max_depth: int = 6, learning_rate: float = 0.1,
                min_samples_leaf: int = 50) -> Pipeline:
    """
    Model Building: Construct a model pipeline with the given hyperparameters.

    Args:
        random_state (int, optional): Random seed for reproducibility. Defaults to 0.
        n_estimators (int, optional): Number of boosting iterations. Defaults to 200.
        max_depth (int, optional): Maximum tree depth. Defaults to 6.
        learning_rate (float, optional): Learning rate for boosting. Defaults to 0.1.
        min_samples_leaf (int, optional): Minimum samples per leaf. Defaults to 50.

    Returns:
        Pipeline: Model pipeline ready for training.
    """
    imp = SimpleImputer(strategy="median")
    gbr = GradientBoostingRegressor(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
    )
    return Pipeline(steps=[
        ("imp", imp),
        ("gbr", gbr)
    ])


@dataclass
class EvaluationMetrics:
    """
    Result of model evaluation.
    """
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> EvaluationMetrics:
    """
    Model Evaluation: Compute evaluation metrics for the given model and dataset.

    Args:
        model (Pipeline): Trained model pipeline to evaluate.
        X (pd.DataFrame): Feature matrix for evaluation.
        y (np.ndarray): True target values for evaluation.

    Returns:
        EvaluationMetrics: A dataclass containing RMSE, MAE, and R-squared results.
    """

    y_pred = model.predict(X)
    return EvaluationMetrics(
        rmse=float(np.sqrt(mean_squared_error(y, y_pred))),
        mae=float(mean_absolute_error(y, y_pred)),
        r2=float(r2_score(y, y_pred)),
    )


class ModelRegistry:
    """
    Model Registry Helper: Encapsulates all MLflow Model Registry operations for a specific model.
    """

    def __init__(self, client: MlflowClient, model_name: str = MODEL_NAME) -> None:
        """
        Initialize the ModelRegistry.

        Args:
            client (MlflowClient): MLflow client.
            model_name (str, optional): Name of the registered model. Defaults to MODEL_NAME.
        """
        try:
            client.get_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            client.create_registered_model(model_name)

        self.client = client
        self.model_name = model_name

    def champion_exists(self) -> bool:
        """
        Check if a champion model exists.

        Returns:
            bool: True if a champion model exists, False otherwise.
        """
        try:
            return self.client.get_model_version_by_alias(self.model_name, "champion") is not None
        except mlflow.exceptions.MlflowException:
            return False

    def load_champion(self) -> Tuple[Any, str]:
        """
        Load the current @champion model. Raises if none exists.

        Returns:
            tuple: Loaded model and its URI.
        """
        uri = f"models:/{self.model_name}@champion"
        return mlflow.pyfunc.load_model(uri), uri

    def register_version(self, model_uri: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register a new model version.

        Args:
            model_uri (str): URI of the model to register.
            tags (Optional[Dict[str, str]], optional): Tags to set on the model version. Defaults to None.

        Returns:
            str: Version number of the registered model.
        """
        mv = mlflow.register_model(model_uri, self.model_name)
        if tags:
            for k, v in tags.items():
                self.client.set_model_version_tag(self.model_name, mv.version, k, v)
        return mv.version

    def promote_to_champion(self, new_version: str, reason: str) -> None:
        """
        Flip @champion alias to new_version, tagging old champion as previous.

        Args:
            new_version (str): Version number to promote.
            reason (str): Reason for promotion.
        """
        # Demote old champion if exists
        try:
            old_mv = self.client.get_model_version_by_alias(self.model_name, "champion")
            old_ver = old_mv.version
            self.client.set_model_version_tag(self.model_name, old_ver, "role", "previous_champion")
            self.client.set_model_version_tag(self.model_name, old_ver, "demoted_at", pd.Timestamp.now().isoformat())
        except mlflow.exceptions.MlflowException:
            pass

        # Promote new
        self.client.set_registered_model_alias(self.model_name, "champion", new_version)
        self.client.set_model_version_tag(self.model_name, new_version, "role", "champion")
        self.client.set_model_version_tag(self.model_name, new_version, "promoted_at", pd.Timestamp.now().isoformat())
        self.client.set_model_version_tag(self.model_name, new_version, "promotion_reason", reason)
