"""
data_loader.py — IPLytics Data Ingestion Module
Generates realistic synthetic IPL ball-by-ball data (2017–2024)
and exposes a clean loading interface.

Schema
------
match_id          : int       — Unique match identifier
season            : int       — IPL season year (2017–2024)
date              : str       — Match date (YYYY-MM-DD)
venue             : str       — Stadium name
batting_team      : str       — Team currently batting
bowling_team      : str       — Team currently bowling
innings           : int       — 1 or 2
over              : int       — 0-indexed over number (0–19)
ball              : int       — Ball within over (1–6)
batsman           : str       — Batsman on strike
bowler            : str       — Bowler delivering the ball
runs_off_bat      : int       — Runs scored off the bat
extras            : int       — Extra runs (wides, no-balls, byes, leg-byes)
total_runs        : int       — runs_off_bat + extras
wicket_kind       : str       — Type of dismissal (or empty)
player_dismissed  : str       — Dismissed player (or empty)
toss_winner       : str       — Team that won the toss
toss_decision     : str       — "bat" or "field"
match_winner      : str       — Team that won the match
target            : int       — Target set by first innings (innings 2 only)
"""

import os
import numpy as np
import pandas as pd
from src.utils import logger, PATHS, set_seed, RANDOM_SEED

set_seed(RANDOM_SEED)

# ─── Constants ───────────────────────────────────────────────────────────────

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals",
    "Sunrisers Hyderabad", "Punjab Kings", "Lucknow Super Giants",
    "Gujarat Titans",
]

VENUES = [
    "Wankhede Stadium", "M.A. Chidambaram Stadium", "Eden Gardens",
    "Arun Jaitley Stadium", "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium", "M. Chinnaswamy Stadium",
    "Punjab Cricket Association Stadium", "Narendra Modi Stadium",
    "Ekana Cricket Stadium",
]

WICKET_KINDS = [
    "caught", "bowled", "lbw", "run out", "stumped",
    "caught and bowled", "hit wicket", "",
]

WICKET_WEIGHTS = [0.04, 0.025, 0.02, 0.01, 0.008, 0.005, 0.002, 0.89]

BATSMEN = [f"Player_{i}" for i in range(1, 51)]
BOWLERS = [f"Bowler_{i}" for i in range(1, 31)]


# ─── Synthetic Data Generation ───────────────────────────────────────────────

def _simulate_innings(
    match_id: int,
    season: int,
    date: str,
    venue: str,
    batting: str,
    bowling: str,
    innings: int,
    toss_winner: str,
    toss_decision: str,
    match_winner: str,
    target: int = 0,
) -> list[dict]:
    """Simulate one full T20 innings ball-by-ball."""

    rows = []
    wickets = 0
    batsman_pool = np.random.choice(BATSMEN, 11, replace=False).tolist()
    bowler_pool  = np.random.choice(BOWLERS, 7, replace=False).tolist()

    # Scoring tendencies: powerplay heavy, death over heavy
    for over in range(20):
        if wickets >= 10:
            break
        bowler = bowler_pool[over % len(bowler_pool)]

        for ball in range(1, 7):
            batsman = batsman_pool[min(wickets, 10)]

            # Phase-based run scoring distribution
            if over < 6:          # Powerplay
                run_probs = [0.28, 0.30, 0.18, 0.08, 0.04, 0.02, 0.10]
            elif over >= 15:      # Death overs
                run_probs = [0.22, 0.25, 0.16, 0.10, 0.07, 0.05, 0.15]
            else:                 # Middle overs
                run_probs = [0.35, 0.30, 0.16, 0.08, 0.03, 0.01, 0.07]

            runs_off_bat = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=run_probs)
            extras = int(np.random.choice([0, 1], p=[0.93, 0.07]))
            total_runs = runs_off_bat + extras

            wicket_kind = np.random.choice(WICKET_KINDS, p=WICKET_WEIGHTS)
            if wickets >= 9:
                wicket_kind = ""      # Last pair can't be dismissed mid-over here

            player_dismissed = batsman if wicket_kind else ""
            if wicket_kind:
                wickets += 1

            rows.append({
                "match_id":         match_id,
                "season":           season,
                "date":             date,
                "venue":            venue,
                "batting_team":     batting,
                "bowling_team":     bowling,
                "innings":          innings,
                "over":             over,
                "ball":             ball,
                "batsman":          batsman,
                "bowler":           bowler,
                "runs_off_bat":     runs_off_bat,
                "extras":           extras,
                "total_runs":       total_runs,
                "wicket_kind":      wicket_kind,
                "player_dismissed": player_dismissed,
                "toss_winner":      toss_winner,
                "toss_decision":    toss_decision,
                "match_winner":     match_winner,
                "target":           target,
            })

    return rows


def generate_synthetic_data(
    n_matches: int = 700,
    output_path: str = PATHS["raw_data"],
) -> pd.DataFrame:
    """
    Generate n_matches IPL matches worth of ball-by-ball data.
    Typical: 700 matches × ~240 balls = ~168 000 rows (scales to 500K+ with 2000+).
    """
    logger.info(f"Generating synthetic data for {n_matches} matches …")
    all_rows: list[dict] = []

    seasons = list(range(2017, 2025))   # 2017-2024
    match_id = 1000

    for _ in range(n_matches):
        season = np.random.choice(seasons)
        month  = np.random.randint(4, 6)
        day    = np.random.randint(1, 28)
        date   = f"{season}-{month:02d}-{day:02d}"
        venue  = np.random.choice(VENUES)

        team_a, team_b = np.random.choice(TEAMS, 2, replace=False)

        toss_winner   = np.random.choice([team_a, team_b])
        toss_decision = np.random.choice(["bat", "field"])

        # Decide batting order
        if toss_decision == "bat":
            bat_first = toss_winner
            bat_second = team_b if toss_winner == team_a else team_a
        else:
            bat_second = toss_winner
            bat_first  = team_b if toss_winner == team_a else team_a

        # Simulate first innings
        inn1 = _simulate_innings(
            match_id, season, date, venue,
            bat_first, bat_second, 1,
            toss_winner, toss_decision,
            match_winner="TBD", target=0,
        )
        first_innings_total = sum(r["total_runs"] for r in inn1) + 1   # +1 = target

        # Decide winner (slight home bias; toss winner wins ~55% of the time)
        toss_advantage   = 0.55 if toss_winner == bat_second else 0.45
        match_winner     = bat_second if np.random.random() < toss_advantage else bat_first

        # Simulate second innings
        inn2 = _simulate_innings(
            match_id, season, date, venue,
            bat_second, bat_first, 2,
            toss_winner, toss_decision,
            match_winner=match_winner, target=first_innings_total,
        )

        # Retroactively set match_winner and target in inn1
        for r in inn1:
            r["match_winner"] = match_winner
            r["target"] = 0   # target is only meaningful in innings 2

        all_rows.extend(inn1)
        all_rows.extend(inn2)
        match_id += 1

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved {len(df):,} rows → {output_path}")
    return df


# ─── Public Loader ────────────────────────────────────────────────────────────

def load_data(path: str = PATHS["raw_data"]) -> pd.DataFrame:
    """Load raw IPL ball-by-ball CSV into a DataFrame."""
    if not os.path.exists(path):
        logger.warning("Raw data not found — generating synthetic dataset …")
        return generate_synthetic_data()

    logger.info(f"Loading data from {path} …")
    df = pd.read_csv(path)
    logger.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def print_schema(df: pd.DataFrame) -> None:
    """Print a data-dictionary style schema."""
    print("\n" + "=" * 60)
    print("  IPLytics — Data Dictionary")
    print("=" * 60)
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        sample  = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
        print(f"  {col:<22} {dtype:<10} unique={nunique:<6} sample={sample}")
    print("=" * 60 + "\n")
