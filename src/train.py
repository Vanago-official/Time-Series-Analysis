"""Model Training and Time-Based Validation module.

Implements TimeSeriesSplit and cutoff-based time split without data leakage.
Trains baseline estimators and Gradient Boosting models, saving results.
"""

import json
import logging
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    LIGHTGBM_PARAMS,
    MODELS_DIR,
    N_CV_SPLITS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SPLIT_DAYS,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "lag_1_duration",
    "lag_2_duration",
    "lag_3_duration",
    "hours_since_prev_alert",
    "rolling_mean_3d",
    "rolling_median_3d",
    "rolling_count_3d",
    "rolling_mean_7d",
    "rolling_median_7d",
    "rolling_count_7d",
    "rolling_mean_14d",
    "rolling_median_14d",
    "rolling_count_14d",
    "expanding_region_mean",
    "active_nationwide",
    "active_neighbors",
    "region_cat",
]


def time_based_split(
    df: pd.DataFrame,
    test_days: int = TEST_SPLIT_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split dataset chronologically based on a cutoff date.

    Train set: everything before cutoff.
    Test set: records in the final `test_days` window.
    Strictly forbids random shuffling to prevent data leakage.
    """
    max_time = df["started_at"].max()
    cutoff_time = max_time - pd.Timedelta(days=test_days)

    train_df = df[df["started_at"] < cutoff_time].copy()
    test_df = df[df["started_at"] >= cutoff_time].copy()

    logger.info("Time-based Split applied:")
    logger.info("  Train Set: %d records [%s -> %s]", len(train_df), train_df["started_at"].min(), train_df["started_at"].max())
    logger.info("  Test Set:  %d records [%s -> %s]", len(test_df), test_df["started_at"].min(), test_df["started_at"].max())
    logger.info("  Cutoff:    %s", cutoff_time)

    return train_df, test_df, cutoff_time


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4)}


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = N_CV_SPLITS,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform TimeSeriesSplit cross-validation on train data."""
    logger.info("Running %d-fold TimeSeriesSplit CV...", n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores: list[dict[str, float]] = []

    model_params = params or LIGHTGBM_PARAMS

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        fold_model = lgb.LGBMRegressor(**model_params)
        fold_model.fit(X_tr, y_tr)
        val_preds = fold_model.predict(X_val)
        val_preds = np.clip(val_preds, 0, None)

        scores = evaluate_predictions(y_val.to_numpy(), val_preds)
        scores["fold"] = fold
        cv_scores.append(scores)
        logger.info("  Fold %d: MAE=%.2f, RMSE=%.2f, R2=%.4f", fold, scores["MAE"], scores["RMSE"], scores["R2"])

    avg_mae = np.mean([s["MAE"] for s in cv_scores])
    avg_rmse = np.mean([s["RMSE"] for s in cv_scores])
    logger.info("CV Average: MAE=%.2f min, RMSE=%.2f min", avg_mae, avg_rmse)

    return {
        "fold_scores": cv_scores,
        "mean_mae": round(float(avg_mae), 3),
        "mean_rmse": round(float(avg_rmse), 3),
    }


def evaluate_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Evaluate simple baseline heuristics on the test set."""
    y_test = test_df[TARGET_COL].to_numpy()

    # Baseline 1: Train Set Median
    median_val = float(train_df[TARGET_COL].median())
    preds_median = np.full_like(y_test, median_val)
    scores_median = evaluate_predictions(y_test, preds_median)

    # Baseline 2: Train Set Mean
    mean_val = float(train_df[TARGET_COL].mean())
    preds_mean = np.full_like(y_test, mean_val)
    scores_mean = evaluate_predictions(y_test, preds_mean)

    # Baseline 3: Rolling 7-day regional mean (from engineered features)
    preds_rolling7 = test_df["rolling_mean_7d"].fillna(median_val).to_numpy()
    scores_rolling7 = evaluate_predictions(y_test, preds_rolling7)

    # Baseline 4: Lag-1 duration (duration of immediate prior alert)
    preds_lag1 = test_df["lag_1_duration"].fillna(median_val).to_numpy()
    scores_lag1 = evaluate_predictions(y_test, preds_lag1)

    baselines = {
        "Global Median Baseline": scores_median,
        "Global Mean Baseline": scores_mean,
        "7-Day Rolling Regional Mean Baseline": scores_rolling7,
        "Lag-1 Prior Alert Baseline": scores_lag1,
    }

    for name, res in baselines.items():
        logger.info("Baseline [%s]: MAE=%.2f min, RMSE=%.2f min", name, res["MAE"], res["RMSE"])

    return baselines


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "lightgbm",
    params: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, dict[str, float]]:
    """Train gradient boosting regressor and predict on test set."""
    logger.info("Training %s regressor...", model_type)

    if model_type == "lightgbm":
        model_params = params or LIGHTGBM_PARAMS
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_type == "hist_gb":
        # Categorical features in HistGradientBoosting need integer encoding or native support
        X_train_hgb = X_train.copy()
        X_test_hgb = X_test.copy()
        if "region_cat" in X_train_hgb.columns:
            X_train_hgb["region_cat"] = X_train_hgb["region_cat"].cat.codes
            X_test_hgb["region_cat"] = X_test_hgb["region_cat"].cat.codes
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            categorical_features=["region_cat"] if "region_cat" in X_train_hgb.columns else None,
        )
        model.fit(X_train_hgb, y_train)
        y_pred = model.predict(X_test_hgb)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Alert duration cannot be negative
    y_pred = np.clip(y_pred, 0, None)
    metrics = evaluate_predictions(y_test.to_numpy(), y_pred)
    logger.info("%s Test Metrics: MAE=%.2f min, RMSE=%.2f min, R2=%.4f", model_type, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    return model, y_pred, metrics


def save_artifacts(
    model: Any,
    features: list[str],
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    """Save trained model, predictions, and summary metrics to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "lightgbm_predictor.joblib"
    joblib.dump({"model": model, "features": features}, model_path)
    logger.info("Saved model artifact to %s", model_path)

    # Save test predictions
    test_preds_df = test_df.copy()
    test_preds_df["predicted_duration_min"] = y_pred
    test_preds_df["error_abs"] = np.abs(test_preds_df[TARGET_COL] - y_pred)
    predictions_path = PROCESSED_DATA_DIR / "test_predictions.csv"
    test_preds_df.to_csv(predictions_path, index=False)
    logger.info("Saved test predictions to %s", predictions_path)

    # Save metrics JSON
    metrics_path = PROCESSED_DATA_DIR / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Saved metrics summary to %s", metrics_path)


def run_training_pipeline(
    df: pd.DataFrame,
    model_type: str = "lightgbm",
    run_cv: bool = True,
) -> dict[str, Any]:
    """Orchestrate time-based split, cross-validation, baseline comparison, and final training."""
    train_df, test_df, cutoff_time = time_based_split(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    # Baselines
    baselines = evaluate_baselines(train_df, test_df)

    # Cross-validation
    cv_results = run_cross_validation(X_train, y_train) if run_cv else {}

    # Final Model Training
    model, y_pred, test_metrics = train_regressor(
        X_train, y_train, X_test, y_test, model_type=model_type
    )

    full_results = {
        "cutoff_time": str(cutoff_time),
        "test_size": len(test_df),
        "train_size": len(train_df),
        "model_type": model_type,
        "test_metrics": test_metrics,
        "baselines": baselines,
        "cv_results": cv_results,
    }

    save_artifacts(model, FEATURE_COLUMNS, test_df, y_pred, full_results)

    return {
        "model": model,
        "test_df": test_df,
        "y_pred": y_pred,
        "results": full_results,
    }
