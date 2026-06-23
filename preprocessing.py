"""
preprocessing.py — IPLytics Data Cleaning & Preprocessing Module

Steps performed
---------------
1. Drop duplicate rows
2. Standardize team names (handle historical renames)
3. Validate numeric columns (clamp outliers)
4. Handle missing values
5. Encode categorical columns needed downstream
6. Persist cleaned DataFrame
"""

import os
import pandas as pd
import numpy as np
from src.utils import logger, PATHS

# ─── Team Name Normalisation Map ────────────────────────────────────────────
# Historical IPL rebrands are mapped to their current names.
TEAM_ALIASES: dict[str, str] = {
    "Delhi Daredevils":            "Delhi Capitals",
    "Kings XI Punjab":             "Punjab Kings",
    "Rising Pune Supergiants":     "Rajasthan Royals",
    "Rising Pune Supergiant":      "Rajasthan Royals",
    "Pune Warriors":               "Rajasthan Royals",
    "Deccan Chargers":             "Sunrisers Hyderabad",
    "Kochi Tuskers Kerala":        "Kerala",
}


# ─── Public API ──────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a cleaned DataFrame.

    Parameters
    ----------
    df   : Raw ball-by-ball DataFrame from data_loader
    save : If True, persist to data/processed/features.csv

    Returns
    -------
    Cleaned DataFrame
    """
    logger.info("Starting data cleaning pipeline …")
    n_raw = len(df)

    df = _drop_duplicates(df)
    df = _normalise_teams(df)
    df = _validate_numerics(df)
    df = _fill_missing(df)
    df = _parse_dates(df)
    df = _add_derived_columns(df)

    logger.success(
        f"Cleaning complete: {n_raw:,} → {len(df):,} rows "
        f"({n_raw - len(df):,} dropped)"
    )

    if save:
        _save(df)

    return df


# ─── Private Helpers ─────────────────────────────────────────────────────────

def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["match_id", "innings", "over", "ball"])
    logger.info(f"  Duplicates removed: {before - len(df):,}")
    return df.reset_index(drop=True)


def _normalise_teams(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["batting_team", "bowling_team", "toss_winner", "match_winner"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_ALIASES)
    logger.info("  Team names normalised")
    return df


def _validate_numerics(df: pd.DataFrame) -> pd.DataFrame:
    # runs off bat: 0–6 (max legal score per ball)
    df["runs_off_bat"] = df["runs_off_bat"].clip(0, 6)
    # extras: reasonable upper bound
    df["extras"] = df["extras"].clip(0, 10)
    # total_runs recomputed from components to fix any mismatches
    df["total_runs"] = df["runs_off_bat"] + df["extras"]
    # over: 0–19, ball: 1–6
    df["over"] = df["over"].clip(0, 19)
    df["ball"] = df["ball"].clip(1, 6)
    logger.info("  Numeric bounds validated")
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = ["wicket_kind", "player_dismissed", "batsman", "bowler", "venue"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    num_cols = ["runs_off_bat", "extras", "total_runs", "target"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    logger.info("  Missing values filled")
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.info("  Dates parsed")
    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight derived columns used by feature engineering."""
    # Ball number in innings (1–120)
    df["ball_num_innings"] = df["over"] * 6 + df["ball"]
    # Is it a wicket delivery?
    df["is_wicket"] = (df["wicket_kind"] != "").astype(int)
    # Is it a boundary?
    df["is_boundary"] = df["runs_off_bat"].isin([4, 6]).astype(int)
    # Is it a dot ball?
    df["is_dot"] = (df["total_runs"] == 0).astype(int)
    logger.info("  Derived columns added (is_wicket, is_boundary, is_dot, ball_num_innings)")
    return df


def _save(df: pd.DataFrame) -> None:
    out = PATHS["processed"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    logger.success(f"  Cleaned data saved → {out}")
