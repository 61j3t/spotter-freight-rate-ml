"""Data loading and the time-based split used for validation.

Train data spans Jan-Oct 2025. The real task predicts Nov-Dec, so every
validation fold here holds out a *later* month than it trains on. That mirrors
"predict the future from the past" instead of leaking it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TARGET = "posted_rate"

# Columns present everywhere (train, validation, December). The reduced model
# is restricted to these because December has nothing else.
SHARED_RAW = ["pickup", "delivery", "distance", "equipment", "weight", "date"]

# Extra columns that exist in train + validation but NOT in December.
EXTRA_RAW = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
]

LABEL_RESIDUAL_THRESHOLD = 0.60


def clean_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Repair deterministic input defects while preserving audit indicators.

    Freight weight cannot be negative. The negative values in these files have
    the same magnitude distribution as valid weights, so they are treated as
    sign-entry errors rather than as distinct observations. Missing and invalid
    zero weights remain missing for fold-fitted model imputers.
    """
    out = df.copy()
    weight = pd.to_numeric(out["weight"], errors="coerce")
    out["weight_was_negative"] = (weight < 0).astype(int)
    out["weight_was_missing"] = weight.isna().astype(int)
    out["weight"] = weight.abs()
    out.loc[out["weight"] == 0, "weight"] = np.nan
    out["date"] = pd.to_datetime(out["date"])
    return out


def corrupted_label_mask(frame: pd.DataFrame, threshold: float = LABEL_RESIDUAL_THRESHOLD) -> pd.Series:
    """Identify the detached multiplicative target-error component.

    A simple equipment + log-distance rate-per-mile baseline exposes a fully
    empty residual band between the ordinary observations and 1.4% of labels
    multiplied or divided by large factors. One robust refit prevents that
    detached component from influencing its own screen.

    This function is only for selecting model-fitting rows. Held-out rows must
    remain untouched when metrics are calculated.
    """
    if TARGET not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)

    distance = pd.to_numeric(frame["distance"], errors="raise").to_numpy(float)
    rate = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
    if np.any(distance <= 0) or np.any(rate <= 0):
        raise ValueError("distance and posted_rate must be positive for label screening")

    design = pd.get_dummies(frame["equipment"], drop_first=True, dtype=float)
    design["log_distance"] = np.log(distance)
    design["intercept"] = 1.0
    matrix = design.to_numpy(float)
    log_rpm = np.log(rate / distance)

    beta, *_ = np.linalg.lstsq(matrix, log_rpm, rcond=None)
    residual = log_rpm - matrix @ beta
    provisional = np.abs(residual - np.median(residual)) < threshold
    beta, *_ = np.linalg.lstsq(matrix[provisional], log_rpm[provisional], rcond=None)
    residual = log_rpm - matrix @ beta
    return pd.Series(np.abs(residual) > threshold, index=frame.index, dtype=bool)


def model_fit_index(frame: pd.DataFrame, candidate_index: pd.Index) -> pd.Index:
    """Return candidate training indices after a candidate-only label screen."""
    candidate = frame.loc[candidate_index]
    return candidate.index[~corrupted_label_mask(candidate)]


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    return clean_inputs(df)


def load_validation() -> pd.DataFrame:
    return clean_inputs(pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"]))


def load_december() -> pd.DataFrame:
    return clean_inputs(pd.read_csv(DATA_DIR / "december_chart_inputs.csv", parse_dates=["date"]))


def time_folds(dates: pd.Series, holdout_months: tuple[int, ...] = (8, 9, 10)):
    """Yield (train_idx, val_idx) expanding-window folds.

    For each holdout month we train on every earlier month and validate on
    that single month. Default holds out Aug, Sep, Oct 2025.
    """
    month = dates.dt.month
    for m in holdout_months:
        train_idx = dates.index[month < m]
        val_idx = dates.index[month == m]
        if len(val_idx) == 0 or len(train_idx) == 0:
            continue
        yield m, train_idx, val_idx
