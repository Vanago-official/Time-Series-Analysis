"""Data Ingestion and Preprocessing module for Ukraine Air Raid Alerts."""

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_OFFICIAL_DATA,
    MAX_DURATION_MINUTES,
    MIN_DURATION_MINUTES,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


def load_raw_data(
    filepath: str | Path | None = None,
    level: str = "oblast",
) -> pd.DataFrame:
    """Load raw air raid alerts dataset from CSV.

    Supports both official_data_*.csv and volunteer_data_*.csv formats.
    """
    path = Path(filepath) if filepath else DEFAULT_OFFICIAL_DATA
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path)

    # Standardize column naming
    if "oblast" in df.columns and "region" not in df.columns:
        df = df.rename(columns={"oblast": "region"})

    if "level" in df.columns and level is not None:
        logger.info("Filtering dataset to level: %s", level)
        df = df[df["level"] == level].copy()

    return df


def clean_data(
    df: pd.DataFrame,
    min_duration: float = MIN_DURATION_MINUTES,
    max_duration: float = MAX_DURATION_MINUTES,
) -> pd.DataFrame:
    """Clean the raw dataframe, compute target variable, and sort strictly by time.

    Steps:
    1. Drop missing critical fields.
    2. Parse started_at and finished_at to UTC datetime.
    3. Drop duplicate entries.
    4. Compute duration in minutes (target variable).
    5. Filter out invalid durations (negative or extreme outliers).
    6. Strictly sort by started_at.
    """
    initial_count = len(df)
    logger.info("Starting data cleaning with %d records", initial_count)

    # 1. Drop missing critical fields
    df = df.dropna(subset=["region", "started_at", "finished_at"]).copy()

    # 2. Parse timestamps
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    # 3. Deduplication
    df = df.drop_duplicates(subset=["region", "started_at", "finished_at"]).copy()
    dedup_count = len(df)
    logger.info("Removed %d duplicate records; remaining: %d", initial_count - dedup_count, dedup_count)

    # 4. Target calculation
    df[TARGET_COL] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60.0

    # 5. Filter invalid durations
    valid_mask = (df[TARGET_COL] >= min_duration) & (df[TARGET_COL] <= max_duration)
    df = df[valid_mask].copy()
    logger.info(
        "Filtered durations [%.1f, %.1f] min: retained %d records",
        min_duration,
        max_duration,
        len(df),
    )

    # 6. Strict chronological sort
    df = df.sort_values(by=["started_at", "region"]).reset_index(drop=True)

    # Sanity checks
    assert not df["started_at"].isnull().any(), "Found null started_at after cleaning"
    assert not df[TARGET_COL].isnull().any(), "Found null target after cleaning"
    assert (df[TARGET_COL] > 0).all(), "Found non-positive durations after cleaning"

    return df


def preprocess_pipeline(
    filepath: str | Path | None = None,
    level: str = "oblast",
) -> pd.DataFrame:
    """Run full ingestion and cleaning pipeline."""
    raw_df = load_raw_data(filepath=filepath, level=level)
    clean_df = clean_data(raw_df)
    logger.info(
        "Preprocessing complete: %d alerts from %s to %s",
        len(clean_df),
        clean_df["started_at"].min().strftime("%Y-%m-%d"),
        clean_df["started_at"].max().strftime("%Y-%m-%d"),
    )
    return clean_df
