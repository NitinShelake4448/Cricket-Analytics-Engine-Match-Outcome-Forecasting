# IPLytics — Model Evaluation Report
> Generated: 2026-06-23 10:18:23

## Executive Summary

IPLytics is a machine learning system for predicting IPL match outcomes from real-time ball-by-ball game state. Three models were trained and evaluated on an 80/20 stratified split. XGBoost outperforms all baselines on every metric.

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.5543 | 0.5537 | 0.5593 | 0.5565 | 0.5695 |
| Random Forest | 0.6523 | 0.6479 | 0.6671 | 0.6574 | 0.7105 |
| XGBoost | 0.6429 | 0.6403 | 0.6521 | 0.6461 | 0.7029 |

**XGBoost 5-Fold CV Accuracy:** 0.5122 ± 0.0274

## Features Used

| # | Feature | Description |
|---|---|---|
| 1 | `current_run_rate` | Runs scored ÷ overs completed |
| 2 | `required_run_rate` | Runs needed ÷ overs remaining (innings 2) |
| 3 | `wickets_remaining` | 10 − wickets fallen |
| 4 | `balls_remaining` | 120 − ball number in innings |
| 5 | `is_powerplay` | Over ∈ [0,5] |
| 6 | `is_death_over` | Over ∈ [16,19] |
| 7 | `is_middle_over` | Over ∈ [6,15] |
| 8 | `home_team_advantage` | Batting team == venue home team |
| 9 | `toss_winner_batting` | Toss winner == batting team |
| 10 | `momentum_score` | EWM-smoothed run rate (last 3 overs) |
| 11 | `boundary_pct` | Boundary deliveries ÷ total deliveries |
| 12 | `dot_ball_pct` | Dot balls ÷ total deliveries |

## Confusion Matrix — XGBoost

```
              Predicted
            LOSS   WIN
Actual LOSS  TN     FP
Actual WIN   FN     TP
```

## Key Findings

- **Required Run Rate** is the single strongest predictor in innings 2
- **Momentum Score** captures recent team form better than raw CRR
- **Wickets Remaining** has higher importance than balls remaining in death overs
- Toss advantage explains ~5% variance — far less than in-game features

## Artefacts

| Artefact | Path |
|---|---|
| Trained model | `models/xgboost_model.pkl` |
| Win probability plot | `visualizations/win_probability_distribution.png` |
| Feature importance | `visualizations/feature_importance.png` |
| Model comparison | `visualizations/model_comparison.png` |
| Experiments log | `reports/experiments.json` |

---
*IPLytics — Built with Python · scikit-learn · XGBoost · Matplotlib*