"""Light Optuna tuning on the strongest models (log framing, full features).

RMSE has already converged near a noise floor across model families, so tuning
is time-boxed: a short search per model, kept honest on the same time-based
folds. We report whether tuning buys anything over the v2 defaults.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from .data import TARGET, load_train, time_folds
from .evaluate import metrics
from .features import fit_schema
from .features_v2 import TargetEncoder, build_static, frame_target

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = Path(__file__).resolve().parent.parent / "results"
TE_COLS = ["lane", "pickup", "delivery"]
SEARCH_SECONDS = 150


def _cv_rmse(make_model, static, train, folds, y, dist, framing="log"):
    fm = []
    for _m, tr, va in folds:
        yf, inv = frame_target(y, dist, framing)
        te = TargetEncoder(TE_COLS).fit(train.loc[tr], yf[tr])
        Xtr = pd.concat([static.loc[tr], te.transform(train.loc[tr])], axis=1)
        Xva = pd.concat([static.loc[va], te.transform(train.loc[va])], axis=1)
        model = make_model()
        model.fit(Xtr, yf[tr])
        pred = inv(model.predict(Xva), dist[va])
        fm.append(metrics(y[va], pred))
    return {k: float(np.mean([f[k] for f in fm])) for k in fm[0]}


def tune() -> None:
    OUT.mkdir(exist_ok=True)
    train = load_train()
    schema = fit_schema(train)
    static = build_static(train, schema, feature_set="full")
    y = train[TARGET].to_numpy(float)
    dist = train["distance"].to_numpy(float)
    folds = list(time_folds(train["date"]))

    spaces = {
        "LightGBM": lambda t: LGBMRegressor(
            n_estimators=t.suggest_int("n_estimators", 300, 1200),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=t.suggest_int("num_leaves", 15, 127),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_samples=t.suggest_int("min_child_samples", 5, 80),
            n_jobs=-1, random_state=42, verbose=-1,
        ),
        "CatBoost": lambda t: CatBoostRegressor(
            iterations=t.suggest_int("iterations", 300, 1200),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=t.suggest_int("depth", 4, 10),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_state=42, verbose=0,
        ),
        "XGBoost": lambda t: XGBRegressor(
            n_estimators=t.suggest_int("n_estimators", 300, 1200),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=t.suggest_int("max_depth", 4, 12),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
            n_jobs=-1, random_state=42,
        ),
    }

    results = []
    for name, build in spaces.items():
        def objective(t):
            return _cv_rmse(lambda: build(t), static, train, folds, y, dist)["RMSE"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, timeout=SEARCH_SECONDS, show_progress_bar=False)
        best = _cv_rmse(lambda: build(optuna.trial.FixedTrial(study.best_params)),
                        static, train, folds, y, dist)
        best.update(model=name, trials=len(study.trials))
        results.append(best)
        print(f"  {name:9s} tuned RMSE={best['RMSE']:.2f} MAE={best['MAE']:.2f} "
              f"MAPE={best['MAPE']:.2f}%  ({len(study.trials)} trials)")
        pd.Series(study.best_params).to_json(OUT / f"best_params_{name}.json")

    pd.DataFrame(results).sort_values("RMSE").to_csv(OUT / "leaderboard_tuned.csv", index=False)
    print(f"\nSaved tuned results + best params to {OUT}")


if __name__ == "__main__":
    tune()
