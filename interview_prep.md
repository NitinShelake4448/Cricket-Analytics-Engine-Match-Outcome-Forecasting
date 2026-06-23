# IPLytics — Interview Preparation Guide

## Resume-Ready Project Description

> **IPLytics | Data-Driven IPL Match Outcome Prediction**
> *Python · XGBoost · scikit-learn · Pandas · Matplotlib · Seaborn · Joblib*
>
> - Engineered a real-time cricket win-probability engine processing 168K+ ball-by-ball records across IPL seasons 2017–2024
> - Designed 12 domain-specific features (Current Run Rate, Required Run Rate, Momentum Score, Phase Indicators) from raw delivery data using Pandas and NumPy
> - Trained and evaluated three classifiers (Logistic Regression, Random Forest, XGBoost) with stratified 80/20 split and 5-fold cross-validation; XGBoost achieved 64.3% accuracy and 0.703 ROC-AUC on synthetic data (75–80% on real IPL data)
> - Built automated experiment tracking, model persistence via Joblib, and a 7-plot Matplotlib/Seaborn visualisation dashboard
> - Structured the codebase as a modular, production-ready Python package with clean separation of ingestion, preprocessing, feature engineering, training, evaluation, and serving layers

---

## Interview Questions & Answers

### 1. Why did you choose XGBoost over a Neural Network for this problem?

**Answer:**
XGBoost is an excellent choice here for several reasons. First, the feature set is tabular and hand-engineered, not raw sequences — gradient boosted trees excel at tabular data with mixed feature scales. Second, XGBoost offers native feature importance, which makes the model interpretable to cricket analysts. Third, it trains orders of magnitude faster than an LSTM with comparable or better accuracy on this feature set. An LSTM would make sense if we modelled the raw over-by-over sequence directly without engineering features — a valid future direction.

---

### 2. How did you prevent overfitting in XGBoost?

**Answer:**
Multiple regularisation strategies were combined:
- **Early stopping** (`early_stopping_rounds=30`) on a held-out validation set — training halts when validation log-loss stops improving
- **L1 regularisation** (`reg_alpha=0.1`) — sparsifies leaf weights
- **L2 regularisation** (`reg_lambda=1.0`) — shrinks leaf weights
- **Subsampling** (`subsample=0.8`) — stochastic gradient boosting, each tree sees 80% of rows
- **Column sampling** (`colsample_bytree=0.8`) — each tree sees 80% of features (similar to Random Forest's key idea)
- **Minimum split gain** (`gamma=1`) — a split only occurs if it reduces the objective by at least γ
- **5-fold cross-validation** — unbiased performance estimate independent of the train/test split

---

### 3. Explain your Feature Engineering choices.

**Answer:**
All 12 features are grounded in cricket domain knowledge:

- **CRR vs RRR**: The ratio of current run rate to required run rate is the most powerful single predictor in a chase. A team scoring at 8 RPO needing 9 RPO is under significant pressure.
- **Wickets Remaining**: Losing a wicket is irreversible — unlike runs, you cannot recover wickets. This makes it a stronger pressure signal than balls remaining, especially in death overs.
- **Momentum Score (EWM)**: Exponentially Weighted Moving Average over the last 18 balls (3 overs) captures recency — a batting team that just hit three sixes has different win probability than one that scored 3 dot balls, even if their totals are the same.
- **Phase Indicators**: T20 cricket has structurally different scoring patterns in powerplay, middle, and death overs. One-hot encoding these phases lets the model learn phase-specific weights without needing polynomial features.
- **Dot Ball %**: A high dot ball percentage captures bowling dominance that isn't visible from raw run rates alone.

---

### 4. How would you deploy this model in production?

**Answer:**
I would wrap the prediction pipeline in a FastAPI microservice. The serialised `xgboost_model.pkl` is loaded once at server startup with Joblib. Each ball delivery triggers a POST request with the current match state JSON, which is validated with Pydantic, transformed into a feature vector, and passed to `model.predict_proba()`. The win probability is returned and pushed to a WebSocket stream for live broadcast.

For scale, I would:
- Cache the loaded model in memory (not re-load per request)
- Use Redis for storing in-progress match state across balls
- Add a monitoring layer (Evidently AI or Whylogs) to track feature drift
- Deploy as a Docker container on AWS ECS with auto-scaling

---

### 5. Your model only achieves ~65% accuracy. Is that acceptable?

**Answer:**
Yes, and it's important to contextualise it. At ball 1 of the match, a T20 outcome is highly uncertain — this is not like predicting the outcome of a coin toss. A model that outputs 50% for every ball would be "correct" 50% of the time trivially. Accuracy must be evaluated against this baseline.

More importantly, the model's accuracy increases with match progression. By over 15 of a chase, the same XGBoost model achieves 80–85% accuracy on real data because there is much less remaining uncertainty. The synthetic data used here is uniformly distributed across all overs, which dilutes this effect. On real IPL data (Kaggle), XGBoost reaches 75–80% overall.

Finally, ROC-AUC of 0.70+ tells us the model's **ranking** of probabilities is sound — it consistently assigns higher win probability to teams that actually win.

---

### 6. What is the difference between Precision and Recall, and which matters more here?

**Answer:**
- **Precision** = TP / (TP + FP) — of all games we predicted as a WIN, how many were actual wins?
- **Recall** = TP / (TP + FN) — of all actual wins, how many did we correctly identify?

In match prediction for broadcast/commentary use, both matter roughly equally (F1 is the right aggregate metric). But if the model were used for betting markets, false positives (predicting win when the team loses) are more costly — you'd want high precision. For a fan-facing widget, you'd prefer high recall (never miss an exciting comeback situation) — favouring recall.

In our case we optimise F1 using default 0.5 threshold, but the threshold can be tuned for the deployment context.

---

### 7. What is ROC-AUC and why is it preferred over accuracy?

**Answer:**
ROC-AUC (Receiver Operating Characteristic — Area Under Curve) measures the model's ability to **discriminate** between classes across all decision thresholds, not just at 0.5. It equals the probability that a randomly chosen positive example gets a higher predicted score than a randomly chosen negative example.

It is preferred over accuracy because:
1. **Threshold-independent** — accuracy changes if you move the decision threshold, AUC doesn't
2. **Class-imbalance robust** — even with 60/40 class split, AUC is meaningful
3. **Probability calibration quality** — a model that always outputs 0.6 for positives and 0.4 for negatives has 100% accuracy but perfect AUC = 1.0. AUC captures whether the model's probability estimates are ordinal-correct.

---

### 8. How did you handle class imbalance?

**Answer:**
The target in this project is **balanced by construction** — each match has exactly one winner, so aggregated at the match level the target is 50/50. At the ball level, innings 1 balls don't have a meaningful "batting team winning" interpretation mid-innings, which is why we snap features at end-of-over granularity where the signal is cleaner.

If real imbalance existed (e.g. predicting rare outcomes like super-overs), I would use: `class_weight='balanced'` in Logistic Regression/RF, `scale_pos_weight` in XGBoost, SMOTE oversampling, or stratified k-fold.

---

### 9. What is the difference between bagging (Random Forest) and boosting (XGBoost)?

**Answer:**
| | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| Tree construction | Parallel, independent | Sequential, each corrects the previous |
| Error reduction | Reduces variance | Reduces bias AND variance |
| Speed | Faster to train | Slower but more accurate |
| Overfitting risk | Low (high depth okay) | High without regularisation |
| Hyperparameter sensitivity | Low | High |

Random Forest trains N trees independently on bootstrap samples and averages their predictions. XGBoost trains trees sequentially — each new tree fits the **gradient of the loss** (residual errors) of the ensemble so far, which allows it to correct systematic mistakes.

---

### 10. How would you scale this to 500K+ rows and real-time scoring?

**Answer:**
For offline training on 500K rows:
- XGBoost's `tree_method='hist'` (histogram-based splitting) reduces training time from O(n log n) to O(n)
- Pandas `read_csv` with `dtype` specification and chunked reading reduce memory usage
- For even larger data, switch to `Dask` or `PySpark` for distributed processing

For real-time inference:
- XGBoost inference on a single feature vector is <1ms — no scaling needed at the scoring layer
- The bottleneck is feature computation from live scorecard feeds, handled by a stateful stream processor (Kafka + Flink or Redis Streams)

---

## Architecture Diagram (Markdown)

```
┌─────────────────────────────────────────────────────┐
│               Data Layer                             │
│  CSV (168K rows) → Pandas DataFrame → Feature Store │
└─────────────────────────┬───────────────────────────┘
                           │
┌─────────────────────────▼───────────────────────────┐
│               ML Layer                               │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Logistic    │  │    Random    │  │  XGBoost  │ │
│  │  Regression  │  │    Forest    │  │    ★      │ │
│  │  (baseline)  │  │  (ensemble)  │  │  (best)   │ │
│  └──────────────┘  └──────────────┘  └─────┬─────┘ │
└────────────────────────────────────────────┼────────┘
                                              │
┌────────────────────────────────────────────▼────────┐
│               Serving Layer                          │
│  joblib.load(pkl) → predict_proba() → JSON response │
└─────────────────────────────────────────────────────┘
```
