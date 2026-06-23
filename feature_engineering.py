"""
feature_engineering.py — IPLytics Cricket-Specific Feature Engineering

Feature Catalogue (12 engineered features)
-------------------------------------------
1.  current_run_rate      — CRR = cumulative_runs / overs_completed
2.  required_run_rate     — RRR = runs_needed / overs_remaining  (innings 2 only)
3.  wickets_remaining     — 10 − wickets_fallen
4.  balls_remaining       — 120 − ball_num_innings
5.  is_powerplay          — over ∈ [0, 5]
6.  is_death_over         — over ∈ [16, 19]
7.  is_middle_over        — over ∈ [6, 15]
8.  home_team_advantage   — batting_team == venue_home_team (proxy via venue name)
9.  toss_winner_batting   — toss_winner == batting_team
10. momentum_score        — exponentially weighted run rate of last 3 overs
11. boundary_pct          — boundaries / total_balls so far (per innings)
12. dot_ball_pct          — dot balls / total_balls so far (per innings)

Target Variable
---------------
batting_team_won  — 1 if match_winner == batting_team, else 0
"""

import pandas as pd
import numpy as np
from src.utils import logger

# ─── Venue → home-city rough mapping (used for home advantage proxy) ─────────
VENUE_HOME: dict[str, str] = {
    "Wankhede Stadium":                     "Mumbai Indians",
    "M.A. Chidambaram Stadium":             "Chennai Super Kings",
    "Eden Gardens":                         "Kolkata Knight Riders",
    "Arun Jaitley Stadium":                 "Delhi Capitals",
    "Rajiv Gandhi International Stadium":   "Sunrisers Hyderabad",
    "Sawai Mansingh Stadium":               "Rajasthan Royals",
    "M. Chinnaswamy Stadium":               "Royal Challengers Bangalore",
    "Punjab Cricket Association Stadium":   "Punjab Kings",
    "Narendra Modi Stadium":                "Gujarat Titans",
    "Ekana Cricket Stadium":                "Lucknow Super Giants",
}


# ─── Public API ──────────────────────────────────────────────────────────────

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all 12 cricket-specific features per ball.
    Operates on the *cleaned* DataFrame produced by preprocessing.clean().

    Returns a row-per-ball DataFrame augmented with engineered columns
    and the binary target `batting_team_won`.
    """
    logger.info("Engineering cricket-specific features …")
    df = df.copy()

    # Sort for cumulative calculations
    df = df.sort_values(["match_id", "innings", "ball_num_innings"]).reset_index(drop=True)

    # Cumulative stats per match-innings
    grp = df.groupby(["match_id", "innings"])
    df["cum_runs"]     = grp["total_runs"].cumsum()
    df["cum_wickets"]  = grp["is_wicket"].cumsum()
    df["cum_balls"]    = grp["ball"].transform(lambda x: range(1, len(x) + 1))
    df["cum_boundaries"] = grp["is_boundary"].cumsum()
    df["cum_dots"]     = grp["is_dot"].cumsum()

    # 1. Current Run Rate
    df["overs_completed"]   = df["ball_num_innings"] / 6
    df["current_run_rate"]  = np.where(
        df["overs_completed"] > 0,
        df["cum_runs"] / df["overs_completed"],
        0.0,
    )

    # 2. Required Run Rate (innings 2 only; NaN for innings 1 → filled with 0)
    df["runs_needed"]      = np.where(
        df["innings"] == 2,
        df["target"] - df["cum_runs"],
        np.nan,
    )
    df["balls_left"]       = 120 - df["ball_num_innings"]
    df["overs_remaining"]  = df["balls_left"] / 6
    df["required_run_rate"] = np.where(
        (df["innings"] == 2) & (df["overs_remaining"] > 0),
        df["runs_needed"] / df["overs_remaining"],
        0.0,
    )
    df["required_run_rate"] = df["required_run_rate"].clip(0, 36)  # cap at 6 per ball

    # 3. Wickets Remaining
    df["wickets_remaining"] = 10 - df["cum_wickets"]

    # 4. Balls Remaining
    df["balls_remaining"] = df["balls_left"].clip(lower=0)

    # 5-7. Phase indicators
    df["is_powerplay"]  = (df["over"] <= 5).astype(int)
    df["is_death_over"] = (df["over"] >= 16).astype(int)
    df["is_middle_over"] = ((df["over"] >= 6) & (df["over"] <= 15)).astype(int)

    # 8. Home Team Advantage
    df["venue_home_team"]     = df["venue"].map(VENUE_HOME).fillna("")
    df["home_team_advantage"] = (df["batting_team"] == df["venue_home_team"]).astype(int)

    # 9. Toss Winner Batting
    df["toss_winner_batting"] = (df["toss_winner"] == df["batting_team"]).astype(int)

    # 10. Momentum Score (EWM run rate over last 3 overs)
    df["momentum_score"] = (
        grp["total_runs"]
        .transform(lambda x: x.ewm(span=18, adjust=False).mean())
    )

    # 11. Boundary Percentage
    df["boundary_pct"] = np.where(
        df["cum_balls"] > 0,
        df["cum_boundaries"] / df["cum_balls"],
        0.0,
    )

    # 12. Dot Ball Percentage
    df["dot_ball_pct"] = np.where(
        df["cum_balls"] > 0,
        df["cum_dots"] / df["cum_balls"],
        0.0,
    )

    # ── Target Variable ──────────────────────────────────────────────────────
    df["batting_team_won"] = (df["match_winner"] == df["batting_team"]).astype(int)

    # ── Encode season as integer ─────────────────────────────────────────────
    df["season"] = df["season"].astype(int)

    logger.success(
        f"Feature engineering complete — {len(FEATURE_COLS)} model features, "
        f"{len(df):,} rows"
    )
    return df


# ─── Feature & Target Columns ────────────────────────────────────────────────

FEATURE_COLS = [
    "innings",
    "over",
    "ball",
    "current_run_rate",
    "required_run_rate",
    "wickets_remaining",
    "balls_remaining",
    "is_powerplay",
    "is_death_over",
    "is_middle_over",
    "home_team_advantage",
    "toss_winner_batting",
    "momentum_score",
    "boundary_pct",
    "dot_ball_pct",
    "season",
]

TARGET_COL = "batting_team_won"


def get_features_and_target(df: pd.DataFrame):
    """Return (X, y) ready for sklearn."""
    # Use end-of-over snapshots for stable features
    snap = (
        df.sort_values("ball_num_innings")
          .groupby(["match_id", "innings", "over"])
          .last()
          .reset_index()
    )
    X = snap[FEATURE_COLS].fillna(0)
    y = snap[TARGET_COL]
    return X, y
