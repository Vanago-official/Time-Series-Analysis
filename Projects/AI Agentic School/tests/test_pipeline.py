"""Unit and Integration Tests for Ukraine Air Raid Alerts Predictor.

Strictly verifies absence of Data Leakage and temporal integrity.
"""

import pandas as pd
import pytest

from src.data_preprocessing import clean_data
from src.feature_engineering import (
    add_lag_and_rolling_features,
    add_spatial_features,
    generate_features,
)
from src.train import FEATURE_COLUMNS, time_based_split, train_regressor


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Create synthetic alerts data with intentional duplicates and known values."""
    data = [
        # Region A alerts
        {"region": "Kyiv City", "started_at": "2024-01-01 10:00:00+00:00", "finished_at": "2024-01-01 11:00:00+00:00"},
        {"region": "Kyiv City", "started_at": "2024-01-01 10:00:00+00:00", "finished_at": "2024-01-01 11:00:00+00:00"},  # duplicate
        {"region": "Kyiv City", "started_at": "2024-01-01 14:00:00+00:00", "finished_at": "2024-01-01 14:45:00+00:00"},
        {"region": "Kyiv City", "started_at": "2024-01-02 08:00:00+00:00", "finished_at": "2024-01-02 09:30:00+00:00"},
        {"region": "Kyiv City", "started_at": "2024-01-03 12:00:00+00:00", "finished_at": "2024-01-03 13:00:00+00:00"},
        {"region": "Kyiv City", "started_at": "2024-01-10 15:00:00+00:00", "finished_at": "2024-01-10 16:20:00+00:00"},
        # Neighboring region (Kyivska oblast)
        {"region": "Kyivska oblast", "started_at": "2024-01-01 10:05:00+00:00", "finished_at": "2024-01-01 11:15:00+00:00"},
        {"region": "Kyivska oblast", "started_at": "2024-01-02 08:10:00+00:00", "finished_at": "2024-01-02 09:40:00+00:00"},
        {"region": "Kyivska oblast", "started_at": "2024-01-10 15:05:00+00:00", "finished_at": "2024-01-10 16:30:00+00:00"},
    ]
    return pd.DataFrame(data)


def test_clean_data_deduplication(sample_raw_df):
    """Test that duplicates are removed and duration_min is calculated accurately."""
    df_clean = clean_data(sample_raw_df)
    assert len(df_clean) == len(sample_raw_df) - 1, "Duplicate row was not dropped"
    assert (df_clean["duration_min"] > 0).all(), "Durations must be strictly positive"
    # Kyiv first alert is 60 min
    first_kyiv = df_clean[df_clean["region"] == "Kyiv City"].iloc[0]
    assert first_kyiv["duration_min"] == 60.0


def test_no_data_leakage_in_lags(sample_raw_df):
    """CRITICAL TEST: Ensure lag_1, lag_2 do not leak the current or future alerts."""
    df_clean = clean_data(sample_raw_df)
    df_features = add_lag_and_rolling_features(df_clean)

    kyiv_alerts = df_features[df_features["region"] == "Kyiv City"].sort_values("started_at").reset_index(drop=True)

    # First alert has no previous alert -> lag_1 must be NaN
    assert pd.isna(kyiv_alerts.loc[0, "lag_1_duration"])

    # Second alert (duration 45 min): lag_1 must be 60 min (duration of the first alert)
    assert kyiv_alerts.loc[1, "lag_1_duration"] == 60.0
    assert kyiv_alerts.loc[1, "duration_min"] == 45.0

    # Third alert (duration 90 min): lag_1 must be 45 min, lag_2 must be 60 min
    assert kyiv_alerts.loc[2, "lag_1_duration"] == 45.0
    assert kyiv_alerts.loc[2, "lag_2_duration"] == 60.0


def test_no_data_leakage_in_rolling(sample_raw_df):
    """CRITICAL TEST: Ensure rolling features strictly exclude current row (closed='left')."""
    df_clean = clean_data(sample_raw_df)
    df_features = add_lag_and_rolling_features(df_clean, rolling_windows=[3])

    kyiv_alerts = df_features[df_features["region"] == "Kyiv City"].sort_values("started_at").reset_index(drop=True)

    # Alert 0: 2024-01-01 10:00 (dur 60). No alerts before it, count must be 0
    assert kyiv_alerts.loc[0, "rolling_count_3d"] == 0.0

    # Alert 1: 2024-01-01 14:00 (dur 45). Exactly 1 alert before it in last 3 days (dur 60)
    assert kyiv_alerts.loc[1, "rolling_count_3d"] == 1.0
    assert kyiv_alerts.loc[1, "rolling_mean_3d"] == 60.0

    # Alert 2: 2024-01-02 08:00 (dur 90). Exactly 2 alerts before it (60 and 45) -> mean = 52.5
    assert kyiv_alerts.loc[2, "rolling_count_3d"] == 2.0
    assert kyiv_alerts.loc[2, "rolling_mean_3d"] == pytest.approx(52.5)


def test_spatial_features(sample_raw_df):
    """Test active alerts count nationwide and in neighboring oblasts at started_at."""
    df_clean = clean_data(sample_raw_df)
    neighbor_map = {"Kyiv City": ["Kyivska oblast"], "Kyivska oblast": ["Kyiv City"]}
    df_spatial = add_spatial_features(df_clean, neighbor_map=neighbor_map)

    # When Kyivska oblast alert starts at 10:05, Kyiv City alert is already active (started 10:00, finished 11:00)
    kyivska_first = df_spatial[(df_spatial["region"] == "Kyivska oblast") & (df_spatial["started_at"].dt.hour == 10)].iloc[0]
    assert kyivska_first["active_neighbors"] == 1
    assert kyivska_first["active_nationwide"] == 1


def test_time_based_split_no_leakage(sample_raw_df):
    """CRITICAL TEST: Verify time-based split preserves strict temporal ordering."""
    df_clean = clean_data(sample_raw_df)
    train_df, test_df, cutoff = time_based_split(df_clean, test_days=5)

    assert train_df["started_at"].max() < cutoff
    assert test_df["started_at"].min() >= cutoff
    assert train_df["started_at"].max() < test_df["started_at"].min()


def test_end_to_end_training_on_synthetic_data(sample_raw_df):
    """Test the training step on engineered features."""
    df_clean = clean_data(sample_raw_df)
    features_df = generate_features(df_clean, include_spatial=True)

    train_df = features_df.iloc[:5].copy()
    test_df = features_df.iloc[5:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["duration_min"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["duration_min"]

    _model, preds, metrics = train_regressor(
        X_train, y_train, X_test, y_test, model_type="lightgbm",
        params={"n_estimators": 10, "min_child_samples": 1, "verbose": -1}
    )

    assert len(preds) == len(test_df)
    assert (preds >= 0).all()
    assert "MAE" in metrics
    assert "RMSE" in metrics
