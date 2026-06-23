"""
main.py — IPLytics Orchestration Entry Point

Run the full end-to-end pipeline:
  python main.py

Steps
-----
  1. Data ingestion (generate or load)
  2. Data cleaning
  3. Feature engineering
  4. Model training (LR, RF, XGBoost)
  5. Evaluation (6 metrics per model)
  6. Visualization dashboard (7 plots)
  7. Model persistence
  8. Prediction demo
  9. Report generation
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import logger, set_seed, ExperimentTracker, PATHS
from src.data_loader import load_data, print_schema
from src.preprocessing import clean
from src.feature_engineering import engineer, get_features_and_target, FEATURE_COLS
from src.train import train_all
from src.predict import load_model, demo_prediction
from src.visualize import generate_all_visualizations
from src.report import generate_report

set_seed(42)


def main():
    logger.info("═" * 60)
    logger.info("  IPLytics — Data-Driven Match Outcome Prediction")
    logger.info("  Starting end-to-end pipeline …")
    logger.info("═" * 60)

    tracker = ExperimentTracker()

    # 1 ── Ingest ──────────────────────────────────────────────────────────────
    logger.info("STEP 1/8 — Data Ingestion")
    df_raw = load_data(PATHS["raw_data"])
    print_schema(df_raw)

    # 2 ── Clean ───────────────────────────────────────────────────────────────
    logger.info("STEP 2/8 — Data Cleaning")
    df_clean = clean(df_raw, save=True)

    # 3 ── Feature Engineering ─────────────────────────────────────────────────
    logger.info("STEP 3/8 — Feature Engineering")
    df_feat = engineer(df_clean)
    X, y = get_features_and_target(df_feat)
    logger.info(f"  Feature matrix: {X.shape}  |  Class balance: {y.mean():.2%} wins")

    # 4-5 ── Training + Evaluation ────────────────────────────────────────────
    logger.info("STEP 4-5/8 — Model Training & Evaluation")
    results, models, split = train_all(X, y, tracker=tracker)
    tracker.save()

    # 6 ── Visualizations ──────────────────────────────────────────────────────
    logger.info("STEP 6/8 — Generating Visualizations")
    xgb_model = models["XGBoost"]
    generate_all_visualizations(
        df=df_feat,
        X_test=split["X_test"],
        y_test=split["y_test"],
        model=xgb_model,
        X=X,
        results=results,
    )

    # 7 ── Prediction Demo ────────────────────────────────────────────────────
    logger.info("STEP 7/8 — Live Prediction Demo")
    saved_model = load_model(PATHS["model"])
    demo_prediction(saved_model)

    # 8 ── Report ──────────────────────────────────────────────────────────────
    logger.info("STEP 8/8 — Generating Model Report")
    generate_report(results, FEATURE_COLS, PATHS["model_report"])

    logger.success("═" * 60)
    logger.success("  Pipeline complete! Artifacts:")
    logger.success(f"  Model       → {PATHS['model']}")
    logger.success(f"  Report      → {PATHS['model_report']}")
    logger.success(f"  Plots       → {PATHS['visualizations']}/")
    logger.success("═" * 60)


if __name__ == "__main__":
    main()
