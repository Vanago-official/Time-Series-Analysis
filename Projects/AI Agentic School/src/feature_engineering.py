"""Feature Engineering module for Ukraine Air Raid Alerts Predictor.

Strictly adheres to Time Series integrity to prevent Data Leakage.
All features are computed using ONLY information available at or before started_at.
"""

import logging

import numpy as np
import pandas as pd

from src.config import NEIGHBORING_OBLASTS, TARGET_COL

logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate calendar and cyclical time features based on started_at."""
    df = df.copy()
    started = df["started_at"]

    df["hour"] = started.dt.hour
    df["dayofweek"] = started.dt.dayofweek
    df["month"] = started.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Generate lag and rolling statistics per region without data leakage.

    Uses closed='left' on rolling windows to ensure the current alert is excluded.
    Uses shift(1), shift(2), shift(3) for strictly prior alert durations.
    """
    if rolling_windows is None:
        rolling_windows = [3, 7, 14]

    df = df.copy()
    dfs_by_region = []

    for region, group in df.groupby("region"):
        group = group.sort_values("started_at").copy()

        # Lag features: duration of 1st, 2nd, and 3rd previous alerts
        group["lag_1_duration"] = group[TARGET_COL].shift(1)
        group["lag_2_duration"] = group[TARGET_COL].shift(2)
        group["lag_3_duration"] = group[TARGET_COL].shift(3)

        # Time elapsed since the previous alert finished (in hours)
        prev_finished = group["finished_at"].shift(1)
        group["hours_since_prev_alert"] = (
            (group["started_at"] - prev_finished).dt.total_seconds() / 3600.0
        )

        # Rolling features over time windows (closed='left' excludes current alert)
        indexed = group.set_index("started_at")
        for days in rolling_windows:
            win_str = f"{days}D"
            mean_series = indexed[TARGET_COL].rolling(win_str, closed="left").mean()
            median_series = indexed[TARGET_COL].rolling(win_str, closed="left").median()
            count_series = indexed[TARGET_COL].rolling(win_str, closed="left").count()

            group[f"rolling_mean_{days}d"] = mean_series.values
            group[f"rolling_median_{days}d"] = median_series.values
            group[f"rolling_count_{days}d"] = count_series.fillna(0).values

        # Expanding regional baseline (historical average up to, but excluding, current alert)
        expanding_mean = group[TARGET_COL].shift(1).expanding(min_periods=1).mean()
        group["expanding_region_mean"] = expanding_mean.values

        # Impute missing rolling means/medians with expanding mean or lag_1
        for days in rolling_windows:
            group[f"rolling_mean_{days}d"] = group[f"rolling_mean_{days}d"].fillna(
                group["expanding_region_mean"]
            )
            group[f"rolling_median_{days}d"] = group[f"rolling_median_{days}d"].fillna(
                group["expanding_region_mean"]
            )

        dfs_by_region.append(group)

    result_df = pd.concat(dfs_by_region).sort_values("started_at").reset_index(drop=True)
    return result_df


def add_spatial_features(
    df: pd.DataFrame,
    neighbor_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Generate spatial features: active alerts nationwide and in neighboring oblasts.

    Calculated at the exact moment started_at using vectorized timeline events.
    """
    if neighbor_map is None:
        neighbor_map = NEIGHBORING_OBLASTS

    df = df.copy().sort_values("started_at").reset_index(drop=True)

    # 1. Total nationwide active alerts (excluding the current one)
    ev_start = pd.DataFrame({"time": df["started_at"], "val": 1})
    ev_end = pd.DataFrame({"time": df["finished_at"], "val": -1})
    events = (
        pd.concat([ev_start, ev_end])
        .sort_values(["time", "val"])
        .reset_index(drop=True)
    )
    events["active_total"] = events["val"].cumsum()
    events = events.drop_duplicates(subset=["time"], keep="last")

    df_active = pd.merge_asof(
        df[["started_at"]],
        events[["time", "active_total"]],
        left_on="started_at",
        right_on="time",
        direction="backward",
    )
    # Subtract 1 so the alert itself is not counted in nationwide background count
    df["active_nationwide"] = np.maximum(0, df_active["active_total"] - 1).fillna(0).astype(int)

    # 2. Per-region active status timelines for neighbor queries
    region_timelines = {}
    for r in df["region"].unique():
        sub = df[df["region"] == r]
        es = pd.DataFrame({"time": sub["started_at"], "val": 1})
        ee = pd.DataFrame({"time": sub["finished_at"], "val": -1})
        ev_r = (
            pd.concat([es, ee])
            .sort_values(["time", "val"])
            .reset_index(drop=True)
        )
        col_name = f"active_{r}"
        ev_r[col_name] = ev_r["val"].cumsum()
        ev_r = ev_r.drop_duplicates(subset=["time"], keep="last")
        region_timelines[r] = (col_name, ev_r[["time", col_name]])

    # 3. Active neighbors count per alert
    df_with_neighbors = []
    for r, group in df.groupby("region"):
        group = group.copy()
        neighbors = neighbor_map.get(r, [])
        neighbor_sum = np.zeros(len(group), dtype=int)

        for n in neighbors:
            if n in region_timelines:
                col_name, tline = region_timelines[n]
                merged = pd.merge_asof(
                    group[["started_at"]],
                    tline,
                    left_on="started_at",
                    right_on="time",
                    direction="backward",
                )
                neighbor_sum += merged[col_name].fillna(0).astype(int).values

        group["active_neighbors"] = neighbor_sum
        df_with_neighbors.append(group)

    result_df = pd.concat(df_with_neighbors).sort_values("started_at").reset_index(drop=True)
    return result_df


def generate_features(
    df: pd.DataFrame,
    include_spatial: bool = True,
) -> pd.DataFrame:
    """Run full feature engineering pipeline.

    Returns DataFrame ready for modeling.
    """
    logger.info("Generating time features...")
    df = add_time_features(df)

    logger.info("Generating lag and rolling features...")
    df = add_lag_and_rolling_features(df)

    if include_spatial:
        logger.info("Generating spatial features...")
        df = add_spatial_features(df)

    # Encode region as category
    df["region_cat"] = df["region"].astype("category")

    logger.info("Feature engineering complete: %d rows, %d columns", len(df), df.shape[1])
    return df
