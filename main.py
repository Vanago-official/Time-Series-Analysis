"""Main execution pipeline for Ukraine Air Raid Alerts Time Series Predictor."""

import argparse
import logging
import time

from src.config import DEFAULT_OFFICIAL_DATA
from src.data_preprocessing import preprocess_pipeline
from src.evaluate import run_evaluation_suite
from src.feature_engineering import generate_features
from src.train import FEATURE_COLUMNS, run_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ukraine Air Raid Alerts Time Series Predictor without Data Leakage."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_OFFICIAL_DATA),
        help="Path to air raid sirens CSV dataset.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="oblast",
        help="Administrative level to analyze (default: oblast).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lightgbm", "hist_gb"],
        default="lightgbm",
        help="Gradient boosting model algorithm.",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Disable spatial features (neighbor status).",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip TimeSeriesSplit cross-validation to speed up execution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    logger.info("=================================================================")
    logger.info("   UKRAINE AIR RAID ALERTS TIME SERIES PREDICTOR (NO LEAKAGE)    ")
    logger.info("=================================================================")
    logger.info("Dataset:    %s", args.data_path)
    logger.info("Level:      %s", args.level)
    logger.info("Model:      %s", args.model)
    logger.info("Spatial:    %s", not args.no_spatial)

    # 1. Data Ingestion & Preprocessing
    logger.info("\n--- STEP 1: DATA INGESTION & PREPROCESSING ---")
    df = preprocess_pipeline(filepath=args.data_path, level=args.level)

    # 2. Feature Engineering
    logger.info("\n--- STEP 2: FEATURE ENGINEERING (STRICT TIME INTEGRITY) ---")
    featured_df = generate_features(df, include_spatial=not args.no_spatial)

    # 3. Model Training & Validation
    logger.info("\n--- STEP 3: MODEL TRAINING & TIME VALIDATION ---")
    train_output = run_training_pipeline(
        featured_df,
        model_type=args.model,
        run_cv=not args.skip_cv,
    )

    # 4. Evaluation & Visualization
    logger.info("\n--- STEP 4: EVALUATION & VISUALIZATION ---")
    generated_plots = run_evaluation_suite(
        model=train_output["model"],
        feature_names=FEATURE_COLUMNS,
        test_df=train_output["test_df"],
        y_pred=train_output["y_pred"],
        results=train_output["results"],
    )

    # Final summary display
    results = train_output["results"]
    test_metrics = results["test_metrics"]
    baselines = results["baselines"]

    elapsed = time.time() - start_time
    logger.info("\n=================================================================")
    logger.info("                      RESULTS SUMMARY                           ")
    logger.info("=================================================================")
    logger.info("Time Cutoff: %s", results["cutoff_time"])
    logger.info("Train Size:  %d records | Test Size: %d records", results["train_size"], results["test_size"])
    logger.info("-----------------------------------------------------------------")
    logger.info("%-38s | %10s | %10s", "Model / Baseline Heuristic", "MAE (min)", "RMSE (min)")
    logger.info("-----------------------------------------------------------------")
    for b_name, b_scores in baselines.items():
        logger.info("%-38s | %10.2f | %10.2f", b_name, b_scores["MAE"], b_scores["RMSE"])
    logger.info("-----------------------------------------------------------------")
    logger.info("%-38s | %10.2f | %10.2f", f"Trained Model ({args.model})", test_metrics["MAE"], test_metrics["RMSE"])
    logger.info("=================================================================")
    logger.info("Generated plots:")
    for p in generated_plots:
        logger.info("  -> %s", p)
    logger.info("Total Pipeline Runtime: %.2f seconds", elapsed)
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
