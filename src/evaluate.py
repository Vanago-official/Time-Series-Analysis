"""Evaluation and Visualization module for Ukraine Air Raid Alerts Predictor.

Generates:
1. Feature Importance plot.
2. Actual vs. Predicted scatter & time-series plots.
3. Baseline vs. Model comparative performance charts.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR, TARGET_COL

logger = logging.getLogger(__name__)

# Set style for publication-ready visualizations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 15,
    save_path: Path | None = None,
) -> Path:
    """Plot feature importance ranking from trained LightGBM model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.warning("Model does not expose feature_importances_, skipping plot.")
        return None

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=fi_df,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    plt.title(f"Top {top_n} Most Important Features (LightGBM)", fontsize=14, weight="bold")
    plt.xlabel("Importance (Split Count / Gain)")
    plt.ylabel("Feature")
    plt.tight_layout()

    out_file = save_path or (PLOTS_DIR / "feature_importance.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved Feature Importance plot to %s", out_file)
    return out_file


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> Path:
    """Generate scatter plot comparing actual vs. predicted durations."""
    plt.figure(figsize=(8, 8))

    max_val = min(300.0, max(np.percentile(y_true, 99), np.percentile(y_pred, 99)))

    sns.scatterplot(
        x=y_true,
        y=y_pred,
        alpha=0.35,
        color="#1f77b4",
        edgecolor=None,
        s=30,
    )

    # 45-degree reference line
    plt.plot([0, max_val], [0, max_val], color="crimson", linestyle="--", lw=2, label="Perfect Prediction")

    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel("Actual Duration (minutes)", fontsize=12)
    plt.ylabel("Predicted Duration (minutes)", fontsize=12)
    plt.title("Actual vs. Predicted Alert Durations (Test Period)", fontsize=14, weight="bold")
    plt.legend()
    plt.tight_layout()

    out_file = save_path or (PLOTS_DIR / "actual_vs_predicted.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved Actual vs Predicted plot to %s", out_file)
    return out_file


def plot_timeseries_sample(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    n_samples: int = 150,
    save_path: Path | None = None,
) -> Path:
    """Plot actual vs predicted durations over a chronological sequence of alerts."""
    sample_df = test_df.tail(n_samples).copy()
    sample_preds = y_pred[-n_samples:]

    plt.figure(figsize=(14, 5))
    x_indices = np.arange(len(sample_df))

    plt.plot(x_indices, sample_df[TARGET_COL].values, label="Actual Duration", color="#d62728", lw=1.5, alpha=0.85)
    plt.plot(x_indices, sample_preds, label="Predicted Duration (LightGBM)", color="#1f77b4", lw=1.8, linestyle="-.")

    plt.title(f"Time Series Forecast: Chronological Sample (Last {n_samples} Alerts)", fontsize=14, weight="bold")
    plt.xlabel("Chronological Alert Index", fontsize=12)
    plt.ylabel("Duration (minutes)", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_file = save_path or (PLOTS_DIR / "timeseries_predictions.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved Time Series Prediction plot to %s", out_file)
    return out_file


def plot_baseline_comparison(
    results: dict[str, Any],
    save_path: Path | None = None,
) -> Path:
    """Generate side-by-side bar chart comparing MAE and RMSE across models & baselines."""
    baselines = results.get("baselines", {})
    test_metrics = results.get("test_metrics", {})
    model_name = "LightGBM Predictor"

    rows = []
    for b_name, b_scores in baselines.items():
        rows.append({"Model": b_name, "Metric": "MAE (min)", "Value": b_scores["MAE"]})
        rows.append({"Model": b_name, "Metric": "RMSE (min)", "Value": b_scores["RMSE"]})

    rows.append({"Model": model_name, "Metric": "MAE (min)", "Value": test_metrics["MAE"]})
    rows.append({"Model": model_name, "Metric": "RMSE (min)", "Value": test_metrics["RMSE"]})

    comp_df = pd.DataFrame(rows)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=comp_df, x="Model", y="Value", hue="Metric", palette=["#2ca02c", "#d62728"])

    plt.title("Model vs. Baselines Performance Comparison (Lower is Better)", fontsize=14, weight="bold")
    plt.xlabel("Model / Baseline Heuristic", fontsize=12)
    plt.ylabel("Error in Minutes", fontsize=12)
    plt.xticks(rotation=20, ha="right")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.1f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    out_file = save_path or (PLOTS_DIR / "baseline_comparison.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved Baseline Comparison plot to %s", out_file)
    return out_file


def run_evaluation_suite(
    model: Any,
    feature_names: list[str],
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    results: dict[str, Any],
) -> list[Path]:
    """Generate all analytical evaluation charts."""
    y_true = test_df[TARGET_COL].to_numpy()

    plots = [
        plot_feature_importance(model, feature_names),
        plot_actual_vs_predicted(y_true, y_pred),
        plot_timeseries_sample(test_df, y_pred),
        plot_baseline_comparison(results),
    ]

    return [p for p in plots if p is not None]
