"""Metrics and the time-based cross-validation leaderboard."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import TARGET, time_folds
from .features import build_features, fit_schema
from .models import model_registry


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 1e-6, None)  # scorer rejects non-positive rates
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / y_true) * 100)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def leaderboard(train: pd.DataFrame, feature_set: str = "full") -> pd.DataFrame:
    """Run every model across the time folds and average the metrics."""
    schema = fit_schema(train)
    X = build_features(train, schema, feature_set=feature_set)
    y = train[TARGET].to_numpy(dtype=float)

    folds = list(time_folds(train["date"]))
    rows = []
    for name, make in model_registry().items():
        fold_metrics = []
        t0 = time.time()
        for _m, tr_idx, va_idx in folds:
            model = make()
            model.fit(X.loc[tr_idx], y[tr_idx])
            pred = model.predict(X.loc[va_idx])
            fold_metrics.append(metrics(y[va_idx], pred))
        agg = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0]}
        agg["secs"] = round(time.time() - t0, 1)
        agg["model"] = name
        rows.append(agg)
        print(
            f"  {name:13s} RMSE={agg['RMSE']:8.2f}  MAE={agg['MAE']:7.2f}  "
            f"MAPE={agg['MAPE']:5.2f}%  R2={agg['R2']:.4f}  ({agg['secs']}s)"
        )

    board = pd.DataFrame(rows).set_index("model").sort_values("RMSE")
    return board[["RMSE", "MAE", "MAPE", "R2", "secs"]]
