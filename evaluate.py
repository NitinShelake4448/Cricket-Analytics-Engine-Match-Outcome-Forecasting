"""
evaluate.py — IPLytics Model Evaluation Module

Computes and returns all six evaluation metrics:
  Accuracy · Precision · Recall · F1 · ROC-AUC · Confusion Matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from src.utils import logger


def evaluate_model(model, X_test, y_test, name: str = "Model") -> dict:
    """
    Evaluate a fitted classifier on test data.

    Returns
    -------
    metrics : dict with keys
        accuracy, precision, recall, f1, roc_auc, confusion_matrix
    """
    y_pred  = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    metrics = {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":          float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    _print_report(name, metrics)
    return metrics


def _print_report(name: str, m: dict) -> None:
    cm = np.array(m["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'─'*50}")
    print(f"  {name} — Evaluation Report")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted")
    print(f"            Loss   Win")
    print(f"  Actual Loss  {tn:4d}  {fp:4d}")
    print(f"  Actual Win   {fn:4d}  {tp:4d}")
    print(f"{'─'*50}\n")
