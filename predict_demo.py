"""Interactive Demo: Predict Air Raid Alert Duration for a new alert."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd

from src.config import MODELS_DIR, NEIGHBORING_OBLASTS

MODEL_PATH = MODELS_DIR / "lightgbm_predictor.joblib"


def predict_alert(
    region: str,
    alert_time: datetime | None = None,
    active_nationwide: int = 5,
    active_neighbors: int = 1,
    hours_since_prev: float = 6.0,
    prev_alert_duration: float = 45.0,
) -> float:
    """Predict the expected duration in minutes for a new alert."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python main.py' first.")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["features"]

    now = alert_time or datetime.now(timezone.utc)
    hour = now.hour
    dayofweek = now.weekday()
    month = now.month
    is_weekend = int(dayofweek >= 5)

    import numpy as np

    sample_dict = {
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        "lag_1_duration": prev_alert_duration,
        "lag_2_duration": prev_alert_duration,
        "lag_3_duration": prev_alert_duration,
        "hours_since_prev_alert": hours_since_prev,
        "rolling_mean_3d": prev_alert_duration,
        "rolling_median_3d": prev_alert_duration,
        "rolling_count_3d": 2.0,
        "rolling_mean_7d": prev_alert_duration,
        "rolling_median_7d": prev_alert_duration,
        "rolling_count_7d": 5.0,
        "rolling_mean_14d": prev_alert_duration,
        "rolling_median_14d": prev_alert_duration,
        "rolling_count_14d": 10.0,
        "expanding_region_mean": prev_alert_duration,
        "active_nationwide": active_nationwide,
        "active_neighbors": active_neighbors,
        "region_cat": region,
    }

    df_sample = pd.DataFrame([sample_dict])
    df_sample["region_cat"] = df_sample["region_cat"].astype("category")

    pred = model.predict(df_sample[feature_cols])[0]
    return max(1.0, float(pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Alert Duration Predictor Demo")
    parser.add_argument("--region", type=str, default="Kyiv City", help="Region name")
    parser.add_argument("--active-nationwide", type=int, default=8, help="Number of other active oblasts")
    parser.add_argument("--active-neighbors", type=int, default=2, help="Number of active neighboring oblasts")
    parser.add_argument("--prev-duration", type=float, default=60.0, help="Duration of previous alert in minutes")
    parser.add_argument("--gap-hours", type=float, default=4.0, help="Hours since last alert finished")

    args = parser.parse_args()

    pred_min = predict_alert(
        region=args.region,
        active_nationwide=args.active_nationwide,
        active_neighbors=args.active_neighbors,
        hours_since_prev=args.gap_hours,
        prev_alert_duration=args.prev_duration,
    )

    print("\n=======================================================")
    print("        ПРОГНОЗ ТРИВАЛОСТІ ПОВІТРЯНОЇ ТРИВОГИ          ")
    print("=======================================================")
    print(f"Регіон:                        {args.region}")
    print(f"Активних тривог по Україні:    {args.active_nationwide}")
    print(f"Активних тривог у сусідів:     {args.active_neighbors}")
    print(f"Тривалість попередньої сирени: {args.prev_duration:.0f} хв")
    print(f"Минуло часу від минулої:       {args.gap_hours:.1f} год")
    print("-------------------------------------------------------")
    print(f"ОЧІКУВАНА ТРИВАЛІСТЬ ТРИВОГИ:  {pred_min:.1f} ХВИЛИН (~{pred_min/60:.1f} год)")
    print("=======================================================\n")
