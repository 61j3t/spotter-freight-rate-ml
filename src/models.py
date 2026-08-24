"""Model roster for the bake-off.

Each entry returns a fresh sklearn-compatible regressor. Models that need
scaled inputs (Ridge, MLP) are wrapped in a StandardScaler pipeline. TabPFN is
intentionally absent here: it runs on Kaggle GPU (see kaggle/), not locally.
"""
from __future__ import annotations

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42


def _scaled(model):
    # median impute (unseen-city NaNs) -> scale -> model
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def model_registry() -> dict:
    return {
        "Ridge": lambda: _scaled(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=8,
            random_state=RANDOM_STATE,
            verbose=0,
        ),
        "MLP": lambda: _scaled(
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )
        ),
    }
