# Spotter — Freight Rate Prediction

Predicts freight `posted_rate` for the Spotter ML Engineer assessment.

Two deliverables are produced:
- `validation_predictions.csv` — rate for all 12,000 validation loads (the graded file).
- The fixed **December chart** (`scorer_results/candidate_december.png`) via the provided `score.py`.

## Approach (short)
- **Two models.** A *full-feature* model for the 12k validation loads (uses
  `market_index`, `quote_signal`, coordinates, etc.) and a *reduced* model for
  December, which only has 6 shared columns.
- **Time-based validation.** Train months are Jan–Oct 2025; every CV fold trains
  on earlier months and validates on a later one (holds out Aug, Sep, Oct).
- **Log-target framing** — predicting `log(rate)` sharply cut proportional error
  (MAPE 8.4% → 5.6%).
- **Bake-off** across Ridge, RandomForest, XGBoost, LightGBM, CatBoost, MLP, and
  TabPFN (TabPFN run separately on Kaggle GPU). Light Optuna tuning on the top 3.
- **Final submission:** an equal-weight blend of the top 3 (CatBoost + Ridge +
  XGBoost), which narrowly beat any single model.

See `REPORT.pdf` for the full write-up and `SPEC.md` for the design decisions.

## Setup & run
Requires [`uv`](https://docs.astral.sh/uv/) (Python 3.11 is pinned).

```bash
uv sync                       # install dependencies
uv run python -m src.run_v1   # v1 baseline leaderboard
uv run python -m src.run_v2   # v2 rich-feature leaderboard
uv run python -m src.tune     # light Optuna tuning (top 3)
uv run python -m src.finalize # select model, write predictions, run scorer
```

`finalize` writes `validation_predictions.csv`, fills
`data/december_chart_inputs.csv`, and runs the provided scorer:

```bash
uv run python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

## Layout
```
src/
  data.py         data loading + time-based folds
  features.py     v1 features (calendar + simple encodings)
  features_v2.py  v2 rich features + leak-safe target encoding + framings
  models.py       model roster
  evaluate.py     metrics + v1 leaderboard
  evaluate_v2.py  v2 leaderboard
  tune.py         Optuna tuning
  finalize.py     best-single-vs-blend selection + final predictions
kaggle/kernel/    self-contained TabPFN GPU kernel
data/             assessment CSVs
results/          leaderboards + tuned params
scorer_results/   candidate_december.png
score.py          provided scorer (unmodified)
```

## TabPFN (Kaggle GPU)
TabPFN needs a GPU; this was developed on a GPU-less Mac, so the TabPFN entry
runs on Kaggle. See `kaggle/kernel/tabpfn_run.py` — a self-contained script that
rebuilds the same features and reports CV metrics + predictions.
