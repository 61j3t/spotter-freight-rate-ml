"""v2 leaderboard: rich features, leak-safe target encoding, target framings."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import TARGET, time_folds
from .evaluate import metrics
from .features import fit_schema
from .features_v2 import TargetEncoder, build_static, frame_target
from .models import model_registry

TE_COLS = ["lane", "pickup", "delivery"]
# RandomForest was slowest and worst in v1 -> dropped from the v2 sweep.
V2_MODELS = {k: v for k, v in model_registry().items() if k != "RandomForest"}


def _fold_matrix(static, df, tr_idx, va_idx, y_framed):
    te = TargetEncoder(TE_COLS).fit(df.loc[tr_idx], y_framed[tr_idx])
    Xtr = pd.concat([static.loc[tr_idx], te.transform(df.loc[tr_idx])], axis=1)
    Xva = pd.concat([static.loc[va_idx], te.transform(df.loc[va_idx])], axis=1)
    return Xtr, Xva


def leaderboard_v2(train: pd.DataFrame, feature_set: str = "full") -> pd.DataFrame:
    schema = fit_schema(train)
    static = build_static(train, schema, feature_set=feature_set)
    y = train[TARGET].to_numpy(dtype=float)
    dist = train["distance"].to_numpy(dtype=float)
    folds = list(time_folds(train["date"]))

    rows = []
    for framing in ("raw", "log", "per_mile"):
        for name, make in V2_MODELS.items():
            fm = []
            t0 = time.time()
            for _m, tr_idx, va_idx in folds:
                y_framed, inv = frame_target(y, dist, framing)
                Xtr, Xva = _fold_matrix(static, train, tr_idx, va_idx, y_framed)
                model = make()
                model.fit(Xtr, y_framed[tr_idx])
                pred = inv(model.predict(Xva), dist[va_idx])
                fm.append(metrics(y[va_idx], pred))
            agg = {k: float(np.mean([f[k] for f in fm])) for k in fm[0]}
            agg.update(model=name, framing=framing, secs=round(time.time() - t0, 1))
            rows.append(agg)
            print(f"  {framing:8s} {name:10s} RMSE={agg['RMSE']:8.2f} "
                  f"MAE={agg['MAE']:7.2f} MAPE={agg['MAPE']:5.2f}% R2={agg['R2']:.4f}")

    board = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return board[["model", "framing", "RMSE", "MAE", "MAPE", "R2", "secs"]]
