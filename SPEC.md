# Spotter Freight Rate ML — Project Spec

## Goal
Predict `posted_rate` for freight loads. Two outputs:
- `validation_predictions.csv` — 12,000 rows. **This is the graded file.**
- December chart — 31 fixed-lane daily predictions (Lexington→Fort Wayne), rendered by `score.py`.

## Environment
- `uv` + Python 3.11, local (MacBook Air M2, 16 GB).
- TabPFN runs on **Kaggle GPU** (CLI logged in as `abdessamedzetroni`); everything else runs local.

## Data facts (from exploration)
- Train = Jan–Oct 2025. Validation = Nov–Dec 2025 (time split by design).
- 64 pickup cities, 64 delivery cities, 4,014 unique lanes.
- Rate range is $57-$25,533. A detached 677-row multiplicative-error component
  is excluded from training only; validation metrics retain every row.
- 292 negative weights are repaired with absolute value and tracked by an audit
  indicator. Missing weights remain available to fold-fitted imputers.
- December lane (Lexington→Fort Wayne) appears 32× in training.
- `validation.csv` has extra columns training also has: `pickup_lat/lon`, `delivery_lat/lon`, `market_index`, `quote_signal`.
- December inputs have only 6 columns: `pickup, delivery, distance, equipment, weight, date`.

## Modeling design
- **Two models:**
  - **Full model** → graded 12k. Uses all columns (incl. market_index, quote_signal, lat/lon).
  - **Reduced model** → December. Uses only the 6 shared columns.
- **Target:** predict total rate; also test a $/mile variant.
- **Validation:** time-based split (train early months, validate the latest).
  Target encoding and label screening are fitted inside CV folds; held-out rows
  are never filtered or altered.
- **Metric:** RMSE decides the winner; also report MAE, MAPE, R².
- **December fix:** calendar + seasonal (sin/cos) features only — never a raw increasing time index — so trees don't flatline on the future month.

## Model roster
Linear, RandomForest, XGBoost, LightGBM, CatBoost, MLP, TabPFN (Kaggle).
Light Optuna tuning on the top 2–3 after the baseline.

## Two passes
1. **v1 baseline** — raw columns + calendar features, full roster at defaults → leaderboard.
2. **v2 deep features** — geo/haversine audit (validate lat/lon vs given distance), market interactions, lane/city target encoding, weight/distance ratios, full calendar, equipment crosses → re-run, compare, prune by feature importance.

## Final submission
Native-categorical LightGBM selected on all-row OOF metrics after comparison
with the equal top-three blend: RMSE 628.7, MAE 113.8, MAPE 5.11%, R² 0.8265.

## Deliverables
- Public repo `spotter-freight-rate-ml` containing assessment code and outputs only.
- `validation_predictions.csv`.
- PDF report (pandoc): split/validation approach + `candidate_december.png`.
- Word-for-word Loom script + on-screen cue sheet (user records).

## Housekeeping
- `validation_predictions_template.csv` link was dead → rebuilt from `validation.csv` load_ids.
- No secrets in the repo.
