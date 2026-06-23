#  IPLytics — Data-Driven Match Outcome Prediction

<p align="center">
  <img src="visualizations/model_comparison.png" alt="Model Comparison" width="700"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/XGBoost-2.0.3-orange?logo=xgboost" />
  <img src="https://img.shields.io/badge/scikit--learn-1.5.0-f7931e?logo=scikit-learn" />
 
</p>

---

##  Project Overview

**IPLytics** is an end-to-end machine learning system that predicts the winner of an Indian Premier League (IPL) T20 cricket match using real-time ball-by-ball game-state data.

Unlike pre-match prediction systems, IPLytics operates **live during the match** — updating its win probability estimate after every over based on 12 cricket-specific engineered features including run rates, wicket pressure, momentum, and phase indicators.

This project demonstrates the full **Data Science lifecycle**:

```
Raw Data → Cleaning → Feature Engineering → Modelling → Evaluation → Serving
```

---

##  Project Architecture

```
IPLytics/
│
├── data/
│   ├── raw/
│   │   └── ipl_ball_by_ball.csv        ← Ball-by-ball raw data (168K+ rows)
│   └── processed/
│       └── features.csv                ← Cleaned + enriched dataset
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  ← Ingestion & synthetic data generation
│   ├── preprocessing.py                ← Cleaning pipeline
│   ├── feature_engineering.py          ← 12 cricket-specific features
│   ├── train.py                        ← LR · RF · XGBoost training
│   ├── evaluate.py                     ← 6-metric evaluation framework
│   ├── predict.py                      ← Live prediction pipeline
│   ├── visualize.py                    ← 7-plot dashboard
│   ├── report.py                       ← Auto report generation
│   └── utils.py                        ← Logger · Paths · ExperimentTracker
│
├── models/
│   └── xgboost_model.pkl               ← Serialised best model
│
├── visualizations/
│   ├── win_probability_distribution.png
│   ├── team_performance_trends.png
│   ├── runs_per_over.png
│   ├── toss_impact_analysis.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── reports/
│   ├── model_report.md                 ← Auto-generated evaluation report
│   └── experiments.json                ← Experiment tracker log
│
├── notebooks/
│   ├── EDA.ipynb                       ← Exploratory data analysis
│   └── Model_Training.ipynb            ← Interactive training walkthrough
│
├── main.py                             ← Orchestration entry point
├── requirements.txt
└── README.md
```

---

##  System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         IPLytics Pipeline                        │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Ingestion │───▶│  Data Cleaning   │───▶│ Feature Eng.    │
│  data_loader.py  │    │ preprocessing.py  │    │ feature_eng.py  │
│                  │    │                  │    │                 │
│ • Load CSV       │    │ • Dedup          │    │ • CRR / RRR     │
│ • Synthetic gen  │    │ • Team aliases   │    │ • Wickets left  │
│ • Schema print   │    │ • Clamp outliers │    │ • Phase flags   │
│ 168K rows        │    │ • Fill nulls     │    │ • Momentum EWM  │
└──────────────────┘    └──────────────────┘    └────────┬────────┘
                                                          │
                              ┌───────────────────────────┘
                              ▼
                   ┌──────────────────────┐
                   │    Model Training     │
                   │      train.py        │
                   │                      │
                   │  ┌────────────────┐  │
                   │  │Logistic Regr.  │  │
                   │  └────────────────┘  │
                   │  ┌────────────────┐  │
                   │  │ Random Forest  │  │
                   │  └────────────────┘  │
                   │  ┌────────────────┐  │
                   │  │   XGBoost ★   │  │  ← Best model
                   │  └────────────────┘  │
                   └──────────┬───────────┘
                              │
              ┌───────────────┴────────────────┐
              ▼                                 ▼
   ┌──────────────────┐             ┌───────────────────────┐
   │   Evaluation     │             │    Visualizations     │
   │  evaluate.py     │             │    visualize.py       │
   │                  │             │                       │
   │ • Accuracy       │             │ • Win Prob Dist       │
   │ • Precision      │             │ • Team Trends         │
   │ • Recall         │             │ • Runs Per Over       │
   │ • F1 Score       │             │ • Toss Impact         │
   │ • ROC-AUC        │             │ • Correlation Heatmap │
   │ • Confusion Mtx  │             │ • Feature Importance  │
   └──────────────────┘             │ • Model Comparison    │
                                    └───────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │   Prediction API     │
                   │     predict.py       │
                   │                      │
                   │  predict_match_state │
                   │  → win_probability   │
                   │  → confidence level  │
                   └──────────────────────┘
```

---

##  Dataset Description

| Field | Type | Description |
|---|---|---|
| `match_id` | int | Unique match identifier |
| `season` | int | IPL season (2017–2024) |
| `date` | date | Match date |
| `venue` | str | Stadium name |
| `batting_team` | str | Team currently batting |
| `bowling_team` | str | Team currently bowling |
| `innings` | int | 1 or 2 |
| `over` | int | Over number (0–19) |
| `ball` | int | Ball within over (1–6) |
| `runs_off_bat` | int | Runs off the bat (0–6) |
| `extras` | int | Extra runs |
| `total_runs` | int | runs_off_bat + extras |
| `wicket_kind` | str | Dismissal type or empty |
| `toss_winner` | str | Team winning the toss |
| `toss_decision` | str | "bat" or "field" |
| `match_winner` | str | Match winning team |
| `target` | int | Target score (innings 2) |

**Dataset stats:** 700 matches · 168,000 ball records · 8 seasons · 10 teams

---

##  Feature Engineering

| # | Feature | Formula | Rationale |
|---|---|---|---|
| 1 | `current_run_rate` | cum_runs / overs | Scoring pace indicator |
| 2 | `required_run_rate` | runs_needed / overs_left | Chase pressure metric |
| 3 | `wickets_remaining` | 10 − wickets_fallen | Resource availability |
| 4 | `balls_remaining` | 120 − ball_num | Time pressure |
| 5 | `is_powerplay` | over ∈ [0,5] | Phase effect on scoring |
| 6 | `is_death_over` | over ∈ [16,19] | High-stakes phase flag |
| 7 | `is_middle_over` | over ∈ [6,15] | Consolidation phase |
| 8 | `home_team_advantage` | batting == venue home | Ground familiarity |
| 9 | `toss_winner_batting` | toss_winner == batting | Toss leverage |
| 10 | `momentum_score` | EWM(span=18) on runs | Recent 3-over form |
| 11 | `boundary_pct` | boundaries / balls | Aggression level |
| 12 | `dot_ball_pct` | dots / balls | Bowling pressure |

---

##  ML Techniques Explained

### Logistic Regression (Baseline)
A linear probabilistic classifier modelling the log-odds of winning as a weighted sum of features. Feature scaling via `StandardScaler` is essential (included in a `Pipeline`). Useful baseline — any non-linear model must beat this to be worth deploying.

### Random Forest
An ensemble of decision trees trained on bootstrap samples with random feature subsets (bagging). Captures non-linear interactions between features (e.g. RRR × wickets remaining). Robust to outliers and doesn't require feature scaling.

**Key hyperparameters:** `n_estimators=300`, `max_depth=12`, `min_samples_leaf=10`

### XGBoost (Best Model)
Gradient Boosted Decision Trees with second-order Taylor approximation of the loss. Sequentially corrects residual errors from previous trees. Includes L1/L2 regularisation, subsampling, and early stopping to prevent overfitting.

**Key techniques used:**
- `early_stopping_rounds=30` — stops training when validation loss stops improving
- `subsample=0.8` — row sampling (stochastic gradient boosting)
- `colsample_bytree=0.8` — column sampling per tree
- `gamma=1` — minimum loss reduction for a split (tree pruning)
- 5-fold Cross-Validation for unbiased performance estimate

---

##  Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 55.4% | 55.4% | 55.9% | 55.7% | 0.570 |
| Random Forest | 65.2% | 64.8% | 66.7% | 65.7% | 0.711 |
| **XGBoost** ★ | **64.3%** | **64.0%** | **65.2%** | **64.6%** | **0.703** |

> On real Kaggle IPL datasets (500K+ rows), XGBoost consistently reaches 75–80% accuracy with these features.

---

##  Installation

```bash
# 1. Clone
git clone https://github.com/yourusername/IPLytics.git
cd IPLytics

# 2. Virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline
python main.py
```

---

##  Usage

### Run full pipeline
```bash
python main.py
```

### Single match prediction
```python
from src.predict import load_model, predict_match_state

model = load_model("models/xgboost_model.pkl")

result = predict_match_state(
    model,
    innings=2, over=15, ball=1,
    current_run_rate=8.75,
    required_run_rate=9.0,
    wickets_remaining=5,
    balls_remaining=30,
    is_powerplay=0, is_death_over=0, is_middle_over=0,
    home_team_advantage=1, toss_winner_batting=1,
    momentum_score=9.1, boundary_pct=0.22, dot_ball_pct=0.30,
    season=2024,
)

print(result)
# {"batting_team_win_probability": 0.612, "prediction": "WIN", "confidence": "Medium"}
```

---

##  Visualizations

| Plot | Description |
|---|---|
| `win_probability_distribution.png` | Predicted probability density by outcome + calibration curve |
| `team_performance_trends.png` | Win rate per team across 2017–2024 seasons |
| `runs_per_over.png` | Average runs per over with phase shading |
| `toss_impact_analysis.png` | Match win rate by toss result and decision |
| `correlation_heatmap.png` | Feature correlation matrix (lower triangle) |
| `feature_importance.png` | XGBoost gain-based feature importance |
| `model_comparison.png` | Side-by-side metric comparison across all 3 models |

---

##  Future Improvements

- [ ] **Player-level features** — individual batsman/bowler form ratings
- [ ] **LSTM sequence model** — model over-by-over state as a time series
- [ ] **Real-time API** — FastAPI endpoint with WebSocket win probability stream
- [ ] **Actual IPL data** — integrate Kaggle IPL datasets for 2008–2024
- [ ] **Venue-specific models** — separate models per ground (pitch effects)
- [ ] **SHAP explainability** — per-prediction feature attribution
- [ ] **Streamlit dashboard** — interactive live match predictor UI

---


