"""Feature construction.

v1 (baseline): raw columns + calendar features + simple encodings.
  - equipment      -> one-hot (3 known categories)
  - pickup/delivery-> ordinal codes (trees split on them fine; id-only, no leak)
  - calendar       -> month, weekday, day-of-month, week-of-year, weekend flag,
                      and smooth seasonal sin/cos of day-of-year so a future
                      month like December stays "in range" for tree models.

`feature_set="full"` adds the extra columns (lat/lon, market_index,
quote_signal); `feature_set="reduced"` keeps only the shared columns and is what
the December model uses.

v2 features (target encoding, geo/haversine, interactions) are added later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EXTRA_RAW

EQUIPMENT_CATEGORIES = ["Dry Van", "Flatbed", "Reefer"]


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    doy = d.dt.dayofyear
    out = pd.DataFrame(index=df.index)
    out["month"] = d.dt.month
    out["weekday"] = d.dt.weekday
    out["day"] = d.dt.day
    out["week"] = d.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (d.dt.weekday >= 5).astype(int)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def fit_schema(train: pd.DataFrame) -> dict:
    """Learn category orderings from training data so codes are consistent
    across train/validation/December. This is id-only encoding, no target."""
    return {
        "pickup": sorted(train["pickup"].astype(str).unique()),
        "delivery": sorted(train["delivery"].astype(str).unique()),
    }


def _ordinal(series: pd.Series, categories: list[str]) -> pd.Series:
    cat = pd.Categorical(series.astype(str), categories=categories)
    codes = cat.codes.astype(float)
    codes[codes < 0] = np.nan  # unseen city -> NaN, trees handle it
    return pd.Series(codes, index=series.index)


def build_features(
    df: pd.DataFrame, schema: dict, feature_set: str = "full"
) -> pd.DataFrame:
    parts = [_calendar(df)]

    base = pd.DataFrame(index=df.index)
    base["distance"] = df["distance"].astype(float)
    base["weight"] = df["weight"].astype(float)
    base["pickup_code"] = _ordinal(df["pickup"], schema["pickup"])
    base["delivery_code"] = _ordinal(df["delivery"], schema["delivery"])
    parts.append(base)

    equip = pd.get_dummies(
        pd.Categorical(df["equipment"], categories=EQUIPMENT_CATEGORIES),
        prefix="equip",
    ).astype(float)
    equip.index = df.index
    parts.append(equip)

    if feature_set == "full":
        extra = df[EXTRA_RAW].astype(float)
        parts.append(extra)

    X = pd.concat(parts, axis=1)
    return X
