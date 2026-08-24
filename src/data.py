"""Data loading and the time-based split used for validation.

Train data spans Jan-Oct 2025. The real task predicts Nov-Dec, so every
validation fold here holds out a *later* month than it trains on. That mirrors
"predict the future from the past" instead of leaking it.
"""
from __future__ import annotations

from pathlib import Path

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


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    return df


def load_validation() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])


def load_december() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "december_chart_inputs.csv", parse_dates=["date"])


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
