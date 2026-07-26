import logging
import numpy as np
import pandas as pd
import pytest
import warnings

from pathlib import Path
from typing import Tuple
from uuid import uuid4

from green_taxi_tip_flow import GreenTaxiTipFlow
from green_taxi_tip_pipeline import engineer_features
from tests.support import make_taxi_df


FeatureXY = Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]


@pytest.fixture(autouse=True)
def suppress_mlflow_warnings():
    """
    Suppress MLflow filesystem deprecation warnings during tests.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The filesystem tracking backend.*will be deprecated",
            category=FutureWarning,
        )
        yield


class SimpleCallRecorder:
    """
    Records calls without using Mock.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def was_called(self) -> bool:
        return len(self.calls) > 0

    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def flow(tmp_path: Path) -> GreenTaxiTipFlow:
    """
    Bare GreenTaxiTipFlow (Metaflow __init__ bypassed).
    """
    uid = uuid4().hex[:8]
    f = object.__new__(GreenTaxiTipFlow)
    f._datastore = None  # Prevent Metaflow __getattr__ recursion
    f.next = SimpleCallRecorder()  # Record next() calls without Mock
    f.tracking_uri = str(tmp_path / "mlruns")  # Use local file-based tracking (no server needed)
    f.experiment_name = "test_experiment"
    f.model_name = "test_model"
    f.ref_path = f"/tmp/ref_{uid}.parquet"
    f.batch_path = f"/tmp/batch_{uid}.parquet"
    f.min_improvement = 0.01
    f.logger = logging.getLogger(f.__class__.__name__)
    f.decision_action = None
    f.integrity_warn = False
    f.candidate_model_uri = None
    f.candidate_rmse_batch = None
    f.candidate_rmse_ref = None
    f.retrain_run_id = None
    f.rmse_champion_on_batch = None
    f.rmse_champion_on_ref = None
    f.champion_model = None
    f.champion_uri = None
    return f


@pytest.fixture
def taxi_ref() -> pd.DataFrame:
    return make_taxi_df(n_rows=200, seed=0)


@pytest.fixture
def taxi_batch() -> pd.DataFrame:
    return make_taxi_df(n_rows=100, seed=1)


@pytest.fixture
def feature_xy() -> FeatureXY:
    """
    Pre-engineered feature matrices and targets.
    """
    ref = make_taxi_df(n_rows=200, seed=0)
    batch = make_taxi_df(n_rows=100, seed=1)
    X_ref, y_ref = engineer_features(ref)
    X_batch, y_batch = engineer_features(batch)
    return X_ref, y_ref, X_batch, y_batch
