import numpy as np
import pandas as pd


def make_taxi_df(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic synthetic taxi dataset with the production schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lpep_pickup_datetime": pd.date_range("2020-01-01", periods=n_rows, freq="h"),
        "lpep_dropoff_datetime": pd.date_range("2020-01-01 00:10:00", periods=n_rows, freq="h"),
        "trip_distance": rng.uniform(0.5, 20, n_rows),
        "fare_amount": rng.uniform(2.5, 100, n_rows),
        "tip_amount": rng.uniform(0, 20, n_rows),
        "PULocationID": rng.integers(1, 265, n_rows),
        "DOLocationID": rng.integers(1, 265, n_rows),
        "passenger_count": rng.integers(1, 6, n_rows).astype(float),
        "payment_type": np.ones(n_rows, dtype=int),
    })
