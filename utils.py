"""
utils.py — IPLytics Utility Functions
Shared helpers for logging, path management, and reproducibility.
"""

import os
import random
import numpy as np
import json
from datetime import datetime

# ─── Reproducibility ────────────────────────────────────────────────────────

RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix all random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Path Helpers ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def path(*parts: str) -> str:
    """Return absolute path relative to project root."""
    return os.path.join(BASE_DIR, *parts)

PATHS = {
    "raw_data":      path("data", "raw", "ipl_ball_by_ball.csv"),
    "processed":     path("data", "processed", "features.csv"),
    "model":         path("models", "xgboost_model.pkl"),
    "model_report":  path("reports", "model_report.md"),
    "visualizations": path("visualizations"),
}


# ─── Logger ──────────────────────────────────────────────────────────────────

class Logger:
    """Minimal structured logger with timestamps."""

    COLORS = {
        "INFO":    "\033[94m",   # blue
        "SUCCESS": "\033[92m",   # green
        "WARNING": "\033[93m",   # yellow
        "ERROR":   "\033[91m",   # red
        "RESET":   "\033[0m",
    }

    def __init__(self, name: str = "IPLytics"):
        self.name = name

    def _log(self, level: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        color = self.COLORS.get(level, "")
        reset = self.COLORS["RESET"]
        print(f"{color}[{ts}] [{self.name}] [{level}] {msg}{reset}")

    def info(self, msg):    self._log("INFO", msg)
    def success(self, msg): self._log("SUCCESS", msg)
    def warning(self, msg): self._log("WARNING", msg)
    def error(self, msg):   self._log("ERROR", msg)


logger = Logger()


# ─── Experiment Tracker ───────────────────────────────────────────────────────

class ExperimentTracker:
    """Lightweight JSON-based experiment tracker."""

    def __init__(self, filepath: str = path("reports", "experiments.json")):
        self.filepath = filepath
        self.experiments = []

    def log(self, name: str, params: dict, metrics: dict) -> None:
        entry = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "metrics": metrics,
        }
        self.experiments.append(entry)
        logger.info(f"Logged experiment: {name}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self.experiments, f, indent=2)
        logger.success(f"Experiments saved → {self.filepath}")
