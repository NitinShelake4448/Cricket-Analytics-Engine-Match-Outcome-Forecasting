"""
predict.py — IPLytics Prediction Pipeline

Provides two interfaces:
  1. predict_match_state()  — single-state prediction from raw game parameters
  2. predict_batch()        — bulk prediction from a feature DataFrame
"""

import joblib
import numpy as np
import pandas as pd
from src.utils import logger, PATHS
from src.feature_engineering import FEATURE_COLS


# ─── Load Model ──────────────────────────────────────────────────────────────

def load_model(path: str = PATHS["model"]):
    """Load a persisted joblib model."""
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


# ─── Single-State Prediction ─────────────────────────────────────────────────

def predict_match_state(
    model,
    innings: int,
    over: int,
    ball: int,
    current_run_rate: float,
    required_run_rate: float,
    wickets_remaining: int,
    balls_remaining: int,
    is_powerplay: int,
    is_death_over: int,
    is_middle_over: int,
    home_team_advantage: int,
    toss_winner_batting: int,
    momentum_score: float,
    boundary_pct: float,
    dot_ball_pct: float,
    season: int = 2024,
) -> dict:
    """
    Predict win probability for the *batting team* given current match state.

    Returns
    -------
    {
        "batting_team_win_probability": float,   # 0.0 – 1.0
        "prediction":                  str,      # "WIN" or "LOSS"
        "confidence":                  str,      # "High" | "Medium" | "Low"
    }
    """
    row = pd.DataFrame([{
        "innings":             innings,
        "over":                over,
        "ball":                ball,
        "current_run_rate":    current_run_rate,
        "required_run_rate":   required_run_rate,
        "wickets_remaining":   wickets_remaining,
        "balls_remaining":     balls_remaining,
        "is_powerplay":        is_powerplay,
        "is_death_over":       is_death_over,
        "is_middle_over":      is_middle_over,
        "home_team_advantage": home_team_advantage,
        "toss_winner_batting": toss_winner_batting,
        "momentum_score":      momentum_score,
        "boundary_pct":        boundary_pct,
        "dot_ball_pct":        dot_ball_pct,
        "season":              season,
    }])

    proba = model.predict_proba(row[FEATURE_COLS])[0][1]
    pred  = "WIN" if proba >= 0.5 else "LOSS"

    confidence = (
        "High"   if proba >= 0.70 or proba <= 0.30 else
        "Medium" if proba >= 0.60 or proba <= 0.40 else
        "Low"
    )

    return {
        "batting_team_win_probability": round(float(proba), 4),
        "prediction":                   pred,
        "confidence":                   confidence,
    }


# ─── Batch Prediction ────────────────────────────────────────────────────────

def predict_batch(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Predict win probabilities for a batch of feature rows.

    Returns the input DataFrame augmented with columns:
        win_probability, prediction
    """
    probas = model.predict_proba(X[FEATURE_COLS])[:, 1]
    result = X.copy()
    result["win_probability"] = probas
    result["prediction"]      = (probas >= 0.5).astype(int)
    return result


# ─── Demo ────────────────────────────────────────────────────────────────────

def demo_prediction(model) -> None:
    """Print example predictions for three representative match states."""
    scenarios = [
        {
            "label":   "Innings 2 — Chasing 180, Over 15, Needing 45 from 30 balls",
            "innings": 2, "over": 15, "ball": 1,
            "current_run_rate": 8.75, "required_run_rate": 9.0,
            "wickets_remaining": 5, "balls_remaining": 30,
            "is_powerplay": 0, "is_death_over": 0, "is_middle_over": 0,
            "home_team_advantage": 1, "toss_winner_batting": 1,
            "momentum_score": 9.1, "boundary_pct": 0.22, "dot_ball_pct": 0.30,
        },
        {
            "label":   "Innings 1 — End of Powerplay, 65/2",
            "innings": 1, "over": 5, "ball": 6,
            "current_run_rate": 10.8, "required_run_rate": 0.0,
            "wickets_remaining": 8, "balls_remaining": 84,
            "is_powerplay": 1, "is_death_over": 0, "is_middle_over": 0,
            "home_team_advantage": 0, "toss_winner_batting": 1,
            "momentum_score": 10.5, "boundary_pct": 0.28, "dot_ball_pct": 0.25,
        },
        {
            "label":   "Innings 2 — Last Over, 15 needed, 2 wickets left",
            "innings": 2, "over": 19, "ball": 1,
            "current_run_rate": 7.8, "required_run_rate": 15.0,
            "wickets_remaining": 2, "balls_remaining": 6,
            "is_powerplay": 0, "is_death_over": 1, "is_middle_over": 0,
            "home_team_advantage": 0, "toss_winner_batting": 0,
            "momentum_score": 5.2, "boundary_pct": 0.18, "dot_ball_pct": 0.40,
        },
    ]

    print("\n" + "═" * 60)
    print("  IPLytics — Live Match Prediction Demo")
    print("═" * 60)
    for s in scenarios:
        kw = {k: v for k, v in s.items() if k != "label"}
        result = predict_match_state(model, season=2024, **kw)
        bar_len = int(result["batting_team_win_probability"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"\n  Scenario: {s['label']}")
        print(f"  [{bar}] {result['batting_team_win_probability']*100:.1f}%")
        print(f"  Prediction: {result['prediction']}  ({result['confidence']} confidence)")
    print("\n" + "═" * 60 + "\n")
