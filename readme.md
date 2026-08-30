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
- **Fold-safe data repair.** Negative weights are corrected, and 677 detached
  labels are excluded from model fitting only; every holdout row is still scored.
- **Log-target framing** — predicting `log(rate)` sharply cut proportional error
  (MAPE 8.4% → 5.1%).
- **Bake-off** across Ridge, RandomForest, XGBoost, LightGBM, CatBoost, MLP, and
  TabPFN (TabPFN run separately on Kaggle GPU). Light Optuna tuning on the top 3.
- **Final submission:** native-categorical LightGBM with training-only lane and
  city frequencies (RMSE 628.7, MAE 113.8, MAPE 5.11%). It beat both every
  numeric single model and the equal top-three blend.

See `REPORT.pdf` for the full write-up and `SPEC.md` for the design decisions.
See `BENCHMARK_NOTES.md` for public-reference provenance and controlled
ablation results.

## Assessment documents

- `REPORT.pdf` — full assessment report.
- `Freight_Rate_Prediction_Minimal.pptx` — concise presentation of the approach and results.
- `Freight_Rate_Prediction_Minimal.pdf` — PDF version of the presentation.

## Setup & run
Requires [`uv`](https://docs.astral.sh/uv/) (Python 3.11 is pinned).
The assessment CSVs are not distributed in this public repository; place them
under `data/` before running the pipeline.

```bash
uv sync                       # install dependencies
uv run python -m src.run_v1   # v1 baseline leaderboard
uv run python -m src.run_v2   # v2 rich-feature leaderboard
uv run python -m src.tune     # light Optuna tuning (top 3)
uv run python -m src.finalize # select model, write predictions, run scorer
uv run python -m unittest discover -s tests -v  # focused regression tests
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
  features_native.py native categorical + training-only frequency features
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
