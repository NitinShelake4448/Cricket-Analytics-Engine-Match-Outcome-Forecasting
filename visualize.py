"""
visualize.py — IPLytics Visualization Dashboard

Produces 7 publication-quality plots saved to /visualizations/:
  1. win_probability_distribution.png
  2. team_performance_trends.png
  3. runs_per_over.png
  4. toss_impact_analysis.png
  5. correlation_heatmap.png
  6. feature_importance.png
  7. model_comparison.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from src.utils import logger, PATHS
from src.feature_engineering import FEATURE_COLS

warnings.filterwarnings("ignore")

# ─── Style Config ─────────────────────────────────────────────────────────────

IPL_PALETTE = {
    "primary":   "#1B4F8A",   # deep blue
    "accent":    "#F5A623",   # IPL gold
    "success":   "#27AE60",   # win green
    "danger":    "#E74C3C",   # loss red
    "neutral":   "#95A5A6",
    "bg":        "#0D1B2A",   # dark navy background
    "card":      "#162032",
    "text":      "#ECF0F1",
}

TEAMS_COLORS = {
    "Mumbai Indians":               "#004BA0",
    "Chennai Super Kings":          "#F8C300",
    "Royal Challengers Bangalore":  "#C8102E",
    "Kolkata Knight Riders":        "#3A225D",
    "Rajasthan Royals":             "#E91E8C",
    "Delhi Capitals":               "#0078BC",
    "Sunrisers Hyderabad":          "#F7A721",
    "Punjab Kings":                 "#AA0000",
    "Lucknow Super Giants":         "#A0D6B4",
    "Gujarat Titans":               "#1C1C1E",
}

def _apply_dark_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":  IPL_PALETTE["bg"],
        "axes.facecolor":    IPL_PALETTE["card"],
        "axes.edgecolor":    "#2C3E50",
        "axes.labelcolor":   IPL_PALETTE["text"],
        "xtick.color":       IPL_PALETTE["text"],
        "ytick.color":       IPL_PALETTE["text"],
        "text.color":        IPL_PALETTE["text"],
        "grid.color":        "#2C3E50",
        "grid.alpha":        0.4,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })

def _save(name: str) -> None:
    out_dir = PATHS["visualizations"]
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{name}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight",
                facecolor=IPL_PALETTE["bg"])
    plt.close()
    logger.success(f"Saved → {fpath}")


# ─── 1. Win Probability Distribution ─────────────────────────────────────────

def plot_win_probability_distribution(model, X_test, y_test) -> None:
    _apply_dark_style()
    probas = model.predict_proba(X_test)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Win Probability Distribution", fontsize=16,
                 color=IPL_PALETTE["accent"], fontweight="bold", y=1.02)

    # KDE plot
    ax = axes[0]
    win_mask  = y_test == 1
    loss_mask = y_test == 0
    ax.hist(probas[win_mask],  bins=40, alpha=0.7, color=IPL_PALETTE["success"],
            label="Actual WIN",  density=True)
    ax.hist(probas[loss_mask], bins=40, alpha=0.7, color=IPL_PALETTE["danger"],
            label="Actual LOSS", density=True)
    ax.axvline(0.5, color=IPL_PALETTE["accent"], linestyle="--", lw=1.5, label="Decision boundary")
    ax.set_xlabel("Predicted Win Probability")
    ax.set_ylabel("Density")
    ax.set_title("Probability Density by Outcome")
    ax.legend(framealpha=0.2)

    # Calibration
    ax2 = axes[1]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    actual_rates = [
        y_test[(probas >= bins[i]) & (probas < bins[i+1])].mean()
        if ((probas >= bins[i]) & (probas < bins[i+1])).sum() > 0 else np.nan
        for i in range(len(bins)-1)
    ]
    ax2.plot([0,1],[0,1], "--", color=IPL_PALETTE["neutral"], lw=1, label="Perfect calibration")
    ax2.plot(bin_centers, actual_rates, "o-", color=IPL_PALETTE["accent"],
             lw=2, ms=8, label="Model calibration")
    ax2.fill_between(bin_centers,
                     [r - 0.05 if r else np.nan for r in actual_rates],
                     [r + 0.05 if r else np.nan for r in actual_rates],
                     color=IPL_PALETTE["accent"], alpha=0.1)
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Positives")
    ax2.set_title("Calibration Curve")
    ax2.legend(framealpha=0.2)

    plt.tight_layout()
    _save("win_probability_distribution")


# ─── 2. Team Performance Trends ───────────────────────────────────────────────

def plot_team_performance_trends(df: pd.DataFrame) -> None:
    _apply_dark_style()
    # Win rate per team per season
    match_level = (
        df.groupby(["match_id", "season", "batting_team", "match_winner"])
          .size().reset_index(name="balls")
    )
    match_level["won"] = (match_level["batting_team"] == match_level["match_winner"]).astype(int)

    team_season = (
        match_level.groupby(["batting_team", "season"])["won"]
        .mean().reset_index()
        .rename(columns={"won": "win_rate"})
    )

    top_teams = (
        team_season.groupby("batting_team")["win_rate"].mean()
        .nlargest(6).index.tolist()
    )
    ts_filtered = team_season[team_season["batting_team"].isin(top_teams)]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Team Win Rate Trends (2017–2024)", fontsize=16,
                 color=IPL_PALETTE["accent"], fontweight="bold")

    for team in top_teams:
        tdata = ts_filtered[ts_filtered["batting_team"] == team].sort_values("season")
        color = TEAMS_COLORS.get(team, IPL_PALETTE["primary"])
        ax.plot(tdata["season"], tdata["win_rate"], "o-", color=color,
                lw=2.5, ms=7, label=team)
        ax.annotate(
            team.split()[0],
            xy=(tdata["season"].iloc[-1], tdata["win_rate"].iloc[-1]),
            xytext=(4, 0), textcoords="offset points",
            fontsize=8, color=color,
        )

    ax.set_xlabel("Season")
    ax.set_ylabel("Win Rate")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.set_xticks(sorted(ts_filtered["season"].unique()))
    ax.legend(loc="lower left", framealpha=0.2, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save("team_performance_trends")


# ─── 3. Runs Per Over ─────────────────────────────────────────────────────────

def plot_runs_per_over(df: pd.DataFrame) -> None:
    _apply_dark_style()
    rpo = df.groupby("over")["total_runs"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Average Runs Per Over Across All Matches", fontsize=16,
                 color=IPL_PALETTE["accent"], fontweight="bold")

    colors = [
        IPL_PALETTE["success"] if o < 6 else
        IPL_PALETTE["primary"] if o < 16 else
        IPL_PALETTE["danger"]
        for o in rpo["over"]
    ]

    bars = ax.bar(rpo["over"], rpo["total_runs"], color=colors, width=0.7, alpha=0.9)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8, color=IPL_PALETTE["text"])

    # Phase shading
    ax.axvspan(-0.5, 5.5,  alpha=0.07, color=IPL_PALETTE["success"], label="Powerplay (0–5)")
    ax.axvspan(5.5, 15.5,  alpha=0.04, color=IPL_PALETTE["primary"],  label="Middle overs (6–15)")
    ax.axvspan(15.5, 19.5, alpha=0.07, color=IPL_PALETTE["danger"],   label="Death overs (16–19)")

    ax.set_xlabel("Over Number")
    ax.set_ylabel("Average Runs")
    ax.set_xticks(range(20))
    ax.legend(framealpha=0.2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save("runs_per_over")


# ─── 4. Toss Impact Analysis ──────────────────────────────────────────────────

def plot_toss_impact(df: pd.DataFrame) -> None:
    _apply_dark_style()
    match_df = df.groupby(["match_id", "toss_winner", "match_winner",
                            "toss_decision"]).size().reset_index(name="c")
    match_df["toss_won_match"] = (match_df["toss_winner"] == match_df["match_winner"]).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Toss Impact Analysis", fontsize=16,
                 color=IPL_PALETTE["accent"], fontweight="bold")

    # Overall toss win %
    ax = axes[0]
    overall = match_df["toss_won_match"].mean()
    ax.barh(
        ["Toss winner", "Toss loser"],
        [overall, 1 - overall],
        color=[IPL_PALETTE["success"], IPL_PALETTE["danger"]],
        height=0.5,
    )
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.set_title("Match Win Rate by Toss Outcome")
    ax.axvline(0.5, color=IPL_PALETTE["accent"], linestyle="--", lw=1)
    for bar, val in zip(ax.patches, [overall, 1 - overall]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=12, color=IPL_PALETTE["text"])

    # Win rate by decision
    ax2 = axes[1]
    by_decision = (
        match_df.groupby("toss_decision")["toss_won_match"].mean()
    )
    ax2.bar(
        by_decision.index, by_decision.values,
        color=[IPL_PALETTE["primary"], IPL_PALETTE["accent"]],
        width=0.4,
    )
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax2.set_title("Win Rate by Toss Decision")
    for i, (idx, val) in enumerate(by_decision.items()):
        ax2.text(i, val + 0.01, f"{val:.1%}", ha="center",
                 fontsize=12, color=IPL_PALETTE["text"])

    plt.tight_layout()
    _save("toss_impact_analysis")


# ─── 5. Correlation Heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap(X: pd.DataFrame) -> None:
    _apply_dark_style()
    corr = X[FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle("Feature Correlation Heatmap", fontsize=16,
                 color=IPL_PALETTE["accent"], fontweight="bold")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor="#2C3E50",
        ax=ax, vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson Correlation Matrix", pad=12)
    plt.tight_layout()
    _save("correlation_heatmap")


# ─── 6. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list[str]) -> None:
    _apply_dark_style()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names   = [feature_names[i] for i in indices]
    sorted_imp     = importances[indices]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle("XGBoost Feature Importance", fontsize=16,
                 color=IPL_PALETTE["accent"], fontweight="bold")

    colors = [
        IPL_PALETTE["accent"] if imp >= sorted_imp[0] * 0.5 else
        IPL_PALETTE["primary"]
        for imp in sorted_imp
    ]
    bars = ax.barh(sorted_names[::-1], sorted_imp[::-1], color=colors[::-1], alpha=0.9)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Gain Importance (XGBoost)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save("feature_importance")


# ─── 7. Model Comparison ──────────────────────────────────────────────────────

def plot_model_comparison(results: dict) -> None:
    _apply_dark_style()
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.25
    colors_list = [IPL_PALETTE["primary"], IPL_PALETTE["accent"], IPL_PALETTE["success"]]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Model Performance Comparison", fontsize=16,
                 color=IPL_PALETTE["accent"], fontweight="bold")

    for i, (name, color) in enumerate(zip(model_names, colors_list)):
        vals = [results[name].get(m, 0) for m in metrics]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, color=IPL_PALETTE["text"])

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n").title() for m in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(framealpha=0.2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.78, color=IPL_PALETTE["danger"], linestyle="--",
               lw=1, label="78% benchmark")
    plt.tight_layout()
    _save("model_comparison")


# ─── Dashboard Entry Point ────────────────────────────────────────────────────

def generate_all_visualizations(df, X_test, y_test, model, X, results) -> None:
    logger.info("Generating visualization dashboard …")
    plot_win_probability_distribution(model, X_test, y_test)
    plot_team_performance_trends(df)
    plot_runs_per_over(df)
    plot_toss_impact(df)
    plot_correlation_heatmap(X)
    plot_feature_importance(model, FEATURE_COLS)
    plot_model_comparison(results)
    logger.success("All 7 visualizations saved to /visualizations/")
