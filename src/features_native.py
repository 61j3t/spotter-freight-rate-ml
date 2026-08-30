"""Native-categorical feature path for LightGBM.

This path complements the shared numeric/target-encoded matrix used by the
other regressors. Pickup, delivery, equipment, and lane remain categorical so
LightGBM can learn category partitions directly. Frequency features are fit on
training rows only for each validation fold.
"""
from __future__ import annotations

import pandas as pd

from .features import EQUIPMENT_CATEGORIES
from .features_v2 import build_static

NATIVE_CATEGORICAL_COLUMNS = [
    "pickup_cat",
    "delivery_cat",
    "equipment_cat",
    "lane_cat",
]


def _lane(df: pd.DataFrame) -> pd.Series:
    return df["pickup"].astype(str) + ">" + df["delivery"].astype(str)


def build_native_static(
    df: pd.DataFrame, schema: dict, feature_set: str = "full"
) -> pd.DataFrame:
    """Build input-only features while preserving native categories."""
    numeric = build_static(df, schema, feature_set).drop(
        columns=["pickup_code", "delivery_code"]
    )
    numeric["day_of_year"] = df["date"].dt.dayofyear
    numeric["days_since_start"] = (
        df["date"] - pd.Timestamp("2025-01-01")
    ).dt.days

    if feature_set == "full":
        distance = df["distance"].astype(float)
        weight = df["weight"].astype(float)
        market = df["market_index"].astype(float)
        quote = df["quote_signal"].astype(float)
        numeric["market_was_missing"] = market.isna().astype(int)
        numeric["quote_was_missing"] = quote.isna().astype(int)
        numeric["weight_x_market"] = weight * market
        numeric["weight_x_quote"] = weight * quote
        numeric["market_x_quote"] = market * quote
        numeric["distance_x_market_x_quote"] = distance * market * quote
        numeric["quote_minus_market"] = quote - market
        numeric["quote_over_market"] = quote / market.clip(lower=1e-6)

    categorical = pd.DataFrame(index=df.index)
    categorical["pickup_cat"] = pd.Categorical(
        df["pickup"].astype(str), categories=schema["pickup"]
    )
    categorical["delivery_cat"] = pd.Categorical(
        df["delivery"].astype(str), categories=schema["delivery"]
    )
    categorical["equipment_cat"] = pd.Categorical(
        df["equipment"].astype(str), categories=EQUIPMENT_CATEGORIES
    )
    categorical["lane_cat"] = pd.Categorical(
        _lane(df), categories=schema["lane"]
    )
    return pd.concat([numeric, categorical], axis=1)


class FrequencyEncoder:
    """Training-only normalized counts for pickup, delivery, and lane."""

    def __init__(self) -> None:
        self.maps: dict[str, pd.Series] = {}

    def fit(self, df: pd.DataFrame) -> "FrequencyEncoder":
        values = {
            "pickup": df["pickup"].astype(str),
            "delivery": df["delivery"].astype(str),
            "lane": _lane(df),
        }
        self.maps = {
            name: series.value_counts(dropna=False) / len(df)
            for name, series in values.items()
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        values = {
            "pickup": df["pickup"].astype(str),
            "delivery": df["delivery"].astype(str),
            "lane": _lane(df),
        }
        out = pd.DataFrame(index=df.index)
        for name, series in values.items():
            out[f"{name}_freq"] = series.map(self.maps[name]).fillna(0).to_numpy()
        return out
