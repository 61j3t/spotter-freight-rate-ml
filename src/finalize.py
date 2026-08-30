"""Finalize: pick best-single vs blend on time-CV, then produce submissions.

Steps:
  1. Build out-of-fold (OOF) predictions for the top local models on the
     Aug/Sep/Oct folds (log framing + leak-safe target encoding).
  2. Compare each single model against an equal-weight blend of the top 3.
  3. Refit the chosen approach on ALL training data and predict:
       - the 12,000 validation loads  (full features)  -> validation_predictions.csv
       - the 31 December days         (reduced features) -> filled december csv
  4. Run the provided score.py to validate outputs and render the chart.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .data import (
    TARGET,
    corrupted_label_mask,
    load_december,
    load_train,
    load_validation,
    model_fit_index,
    time_folds,
)
from .evaluate import metrics
from .features import fit_schema
from .features_v2 import TargetEncoder, build_static, frame_target

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
TE_COLS = ["lane", "pickup", "delivery"]
DECEMBER_OUTPUT_COLUMNS = [
    "pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"
]


def _params(name: str) -> dict:
    p = OUT / f"best_params_{name}.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    # pd.Series JSON coerced ints to floats; restore integral values to int.
    return {k: (int(v) if isinstance(v, float) and v.is_integer() else v)
            for k, v in raw.items()}


def candidates() -> dict:
    return {
        "Ridge": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
        ),
        "CatBoost": lambda: CatBoostRegressor(
            **{**_params("CatBoost"), "random_state": 42, "verbose": 0}
        ),
        "XGBoost": lambda: XGBRegressor(
            **{**_params("XGBoost"), "random_state": 42, "n_jobs": -1}
        ),
        "LightGBM": lambda: LGBMRegressor(
            **{**_params("LightGBM"), "random_state": 42, "n_jobs": -1, "verbose": -1}
        ),
    }


def _matrix(static, df, tr, va, yframed):
    te = TargetEncoder(TE_COLS).fit(df.loc[tr], yframed[tr])
    Xtr = pd.concat([static.loc[tr], te.transform(df.loc[tr])], axis=1)
    Xva = pd.concat([static.loc[va], te.transform(df.loc[va])], axis=1)
    return Xtr, Xva


def oof_selection(train, feature_set="full", framing="log"):
    schema = fit_schema(train)
    static = build_static(train, schema, feature_set)
    y = train[TARGET].to_numpy(float)
    dist = train["distance"].to_numpy(float)
    folds = list(time_folds(train["date"]))

    oof = {name: np.full(len(train), np.nan) for name in candidates()}
    mask = np.zeros(len(train), dtype=bool)
    pos = {ix: i for i, ix in enumerate(train.index)}
    for _m, tr, va in folds:
        fit_idx = model_fit_index(train, tr)
        yf, inv = frame_target(y, dist, framing)
        Xtr, Xva = _matrix(static, train, fit_idx, va, yf)
        va_pos = [pos[i] for i in va]
        mask[va_pos] = True
        for name, make in candidates().items():
            model = make(); model.fit(Xtr, yf[fit_idx])
            oof[name][va_pos] = inv(model.predict(Xva), dist[va.to_numpy()])

    yv = y[mask]
    board = []
    for name, pred in oof.items():
        board.append({"cand": name, **metrics(yv, pred[mask])})
    single = pd.DataFrame(board).sort_values("RMSE").reset_index(drop=True)

    top3 = single["cand"].head(3).tolist()
    blend_pred = np.mean([oof[n][mask] for n in top3], axis=0)
    blend = {"cand": f"Blend({'+'.join(top3)})", **metrics(yv, blend_pred)}

    table = pd.concat([single, pd.DataFrame([blend])], ignore_index=True).sort_values("RMSE")
    return table.reset_index(drop=True), top3


def _fit_predict(train, target_df, feature_set, framing, model_names):
    """Refit given model(s) on ALL train, predict target_df; average if many."""
    schema = fit_schema(train)
    static_tr = build_static(train, schema, feature_set)
    static_te = build_static(target_df, schema, feature_set)
    y = train[TARGET].to_numpy(float)
    dist_tr = train["distance"].to_numpy(float)
    dist_te = target_df["distance"].to_numpy(float)
    yf, inv = frame_target(y, dist_tr, framing)

    fit_idx = train.index[~corrupted_label_mask(train)]
    te = TargetEncoder(TE_COLS).fit(train.loc[fit_idx], yf[fit_idx])
    Xtr = pd.concat([static_tr.loc[fit_idx], te.transform(train.loc[fit_idx])], axis=1)
    Xte = pd.concat([static_te, te.transform(target_df)], axis=1)

    preds = []
    reg = candidates()
    for name in model_names:
        model = reg[name](); model.fit(Xtr, yf[fit_idx])
        preds.append(inv(model.predict(Xte), dist_te))
    return np.clip(np.mean(preds, axis=0), 1e-6, None)


def _december_output_frame(december: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Restore the scorer's fixed schema after internal audit preprocessing."""
    out = december[DECEMBER_OUTPUT_COLUMNS[:-1]].copy()
    out["predicted_rate"] = predictions
    return out[DECEMBER_OUTPUT_COLUMNS]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train = load_train()

    print("== Final selection (OOF, full features) ==")
    table, top3 = oof_selection(train, feature_set="full")
    print(table.to_string(index=False))
    table.to_csv(OUT / "final_selection.csv", index=False)

    best = table.iloc[0]["cand"]
    if best.startswith("Blend"):
        chosen = top3
        print(f"\nChosen: BLEND of {top3}")
    else:
        chosen = [best]
        print(f"\nChosen: single model {best}")

    # 12k validation predictions (full features)
    valid = load_validation()
    vpred = _fit_predict(train, valid, "full", "log", chosen)
    sub = pd.DataFrame({"load_id": valid["load_id"], "predicted_rate": vpred})
    sub.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"\nWrote validation_predictions.csv ({len(sub):,} rows, "
          f"mean=${vpred.mean():.0f})")

    # December (reduced features) — CatBoost-log is the reduced champion
    december = load_december()
    dpred = _fit_predict(train, december, "reduced", "log", ["CatBoost"])
    dec_out = _december_output_frame(december, dpred)
    dec_path = ROOT / "data" / "december_chart_inputs.csv"
    dec_out.to_csv(dec_path, index=False)
    print(f"Filled {dec_path.name} (mean=${dpred.mean():.0f})")

    # run provided scorer
    print("\n== Running provided score.py ==")
    r = subprocess.run(
        [sys.executable, "score.py", "--predictions", "validation_predictions.csv",
         "--december-predictions", "data/december_chart_inputs.csv"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(r.stdout or r.stderr)


if __name__ == "__main__":
    main()
