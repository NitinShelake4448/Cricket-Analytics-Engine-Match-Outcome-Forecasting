"""
train.py — IPLytics Model Training Module

Models trained
--------------
1. Logistic Regression   (sklearn)
2. Random Forest         (sklearn)
3. XGBoost               (xgboost via sklearn API)

All models use a shared 80/20 stratified train-test split.
Hyperparameters are lightly tuned; XGBoost uses early stopping.
Trained models are persisted via joblib.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.utils import logger, PATHS, RANDOM_SEED, ExperimentTracker
from src.evaluate import evaluate_model


# ─── Hyperparameters ─────────────────────────────────────────────────────────

LR_PARAMS = {
    "max_iter":  1000,
    "C":         1.0,
    "solver":    "lbfgs",
    "random_state": RANDOM_SEED,
}

RF_PARAMS = {
    "n_estimators":   300,
    "max_depth":      12,
    "min_samples_leaf": 10,
    "n_jobs":         -1,
    "random_state":   RANDOM_SEED,
}

XGB_PARAMS = {
    "n_estimators":       500,
    "learning_rate":      0.05,
    "max_depth":          6,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "gamma":              1,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "eval_metric":        "logloss",
    "early_stopping_rounds": 30,
    "use_label_encoder":  False,
    "random_state":       RANDOM_SEED,
    "verbosity":          0,
}


# ─── Public API ──────────────────────────────────────────────────────────────

def train_all(X, y, tracker: ExperimentTracker | None = None):
    """
    Train all three models, evaluate, and persist XGBoost.

    Returns
    -------
    results : dict[str, dict]   — per-model evaluation metrics
    models  : dict[str, object] — fitted model objects
    split   : dict              — X_train, X_test, y_train, y_test
    """
    logger.info("Splitting data 80/20 (stratified) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    logger.info(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    results, models = {}, {}

    # 1 ── Logistic Regression ────────────────────────────────────────────────
    logger.info("Training Logistic Regression …")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**LR_PARAMS)),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_pipeline, X_test, y_test, name="Logistic Regression")
    results["Logistic Regression"] = lr_metrics
    models["Logistic Regression"]  = lr_pipeline

    if tracker:
        tracker.log("Logistic Regression", LR_PARAMS, lr_metrics)

    # 2 ── Random Forest ──────────────────────────────────────────────────────
    logger.info("Training Random Forest …")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, name="Random Forest")
    results["Random Forest"] = rf_metrics
    models["Random Forest"]  = rf

    if tracker:
        tracker.log("Random Forest", RF_PARAMS, rf_metrics)

    # 3 ── XGBoost ────────────────────────────────────────────────────────────
    logger.info("Training XGBoost …")
    # Pop early_stopping_rounds from params dict; pass separately
    xgb_init_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    xgb_init_params["early_stopping_rounds"] = XGB_PARAMS["early_stopping_rounds"]
    xgb = XGBClassifier(**xgb_init_params)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_metrics = evaluate_model(xgb, X_test, y_test, name="XGBoost")
    results["XGBoost"] = xgb_metrics
    models["XGBoost"]  = xgb

    if tracker:
        tracker.log("XGBoost", XGB_PARAMS, xgb_metrics)

    # ── Cross-Validation (XGBoost) ───────────────────────────────────────────
    logger.info("Running 5-fold cross-validation on XGBoost …")
    cv_params = {k: v for k, v in xgb_init_params.items() if k != "early_stopping_rounds"}
    cv_scores = cross_val_score(
        XGBClassifier(**cv_params),
        X, y,
        cv=5, scoring="accuracy", n_jobs=-1,
    )
    logger.info(
        f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )
    results["XGBoost"]["cv_mean"] = float(cv_scores.mean())
    results["XGBoost"]["cv_std"]  = float(cv_scores.std())

    # ── Persist Best Model ───────────────────────────────────────────────────
    _save_model(xgb, PATHS["model"])

    split = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }

    _print_leaderboard(results)
    return results, models, split


# ─── Private Helpers ─────────────────────────────────────────────────────────

def _save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.success(f"XGBoost model persisted → {path}")


def _print_leaderboard(results: dict) -> None:
    print("\n" + "═" * 58)
    print("  IPLytics — Model Leaderboard")
    print("═" * 58)
    header = f"  {'Model':<25} {'Acc':>6} {'AUC':>6} {'F1':>6}"
    print(header)
    print("  " + "-" * 54)
    for name, m in sorted(results.items(), key=lambda x: -x[1]["roc_auc"]):
        print(
            f"  {name:<25} "
            f"{m['accuracy']:>6.4f} "
            f"{m['roc_auc']:>6.4f} "
            f"{m['f1']:>6.4f}"
        )
    print("═" * 58 + "\n")
