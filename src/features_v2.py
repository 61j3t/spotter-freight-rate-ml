"""v2 rich features + leak-safe target encoding + target framings.

Static features (built once): everything from v1 plus
  - log_distance, weight_per_mile, weight, distance buckets
  - interactions: distance x equipment  (and, full only,
    distance x market_index, distance x quote_signal, detour ratio)
  - US federal holiday flag

Target encoding (fit inside each CV fold only, never on the val rows):
  - lane (pickup>delivery), pickup city, delivery city
  Smoothed toward the global mean so rare lanes don't overfit.

Target framings (Q21): raw / log / per_mile. per_mile predicts $-per-mile then
multiplies by distance, which suits the near-linear distance relationship.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from .features import EQUIPMENT_CATEGORIES, _calendar, _ordinal

US_HOLIDAYS = holidays.US(years=[2025])


def build_static(df: pd.DataFrame, schema: dict, feature_set: str = "full") -> pd.DataFrame:
    parts = [_calendar(df)]
    b = pd.DataFrame(index=df.index)
    dist = df["distance"].astype(float)
    wt = df["weight"].astype(float)
    b["distance"] = dist
    b["log_distance"] = np.log1p(dist)
    b["weight"] = wt
    b["weight_per_mile"] = wt / dist.clip(lower=1)
    b["pickup_code"] = _ordinal(df["pickup"], schema["pickup"])
    b["delivery_code"] = _ordinal(df["delivery"], schema["delivery"])
    b["is_holiday"] = df["date"].dt.date.map(lambda d: d in US_HOLIDAYS).astype(int)
    parts.append(b)

    equip = pd.get_dummies(
        pd.Categorical(df["equipment"], categories=EQUIPMENT_CATEGORIES), prefix="equip"
    ).astype(float)
    equip.index = df.index
    parts.append(equip)

    inter = pd.DataFrame(index=df.index)
    for cat in EQUIPMENT_CATEGORIES:
        inter[f"dist_x_{cat.replace(' ','')}"] = dist * (df["equipment"] == cat).astype(float)
    parts.append(inter)

    if feature_set == "full":
        ex = pd.DataFrame(index=df.index)
        ex["market_index"] = df["market_index"].astype(float)
        ex["quote_signal"] = df["quote_signal"].astype(float)
        ex["dist_x_market"] = dist * ex["market_index"]
        ex["dist_x_quote"] = dist * ex["quote_signal"]
        # detour ratio: given distance vs straight-line (synthetic but consistent)
        R = 3958.8
        p = np.pi / 180
        a = (np.sin((df.delivery_lat - df.pickup_lat) * p / 2) ** 2
             + np.cos(df.pickup_lat * p) * np.cos(df.delivery_lat * p)
             * np.sin((df.delivery_lon - df.pickup_lon) * p / 2) ** 2)
        hav = 2 * R * np.arcsin(np.sqrt(a.clip(0, 1)))
        ex["detour_ratio"] = dist / hav.clip(lower=1)
        parts.append(ex)

    return pd.concat(parts, axis=1)


class TargetEncoder:
    """Smoothed mean target encoding. Fit on TRAIN rows only."""

    def __init__(self, cols: list[str], smoothing: float = 50.0):
        self.cols = cols
        self.smoothing = smoothing
        self.maps: dict[str, pd.Series] = {}
        self.global_mean = 0.0

    @staticmethod
    def _lane(df: pd.DataFrame) -> pd.Series:
        return df["pickup"].astype(str) + ">" + df["delivery"].astype(str)

    def _col_values(self, df: pd.DataFrame, col: str) -> pd.Series:
        return self._lane(df) if col == "lane" else df[col].astype(str)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "TargetEncoder":
        self.global_mean = float(np.mean(y))
        tmp = pd.DataFrame({"y": y})
        for col in self.cols:
            tmp[col] = self._col_values(df, col).to_numpy()
            stats = tmp.groupby(col)["y"].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + self.global_mean * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.maps[col] = smooth
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.cols:
            vals = self._col_values(df, col)
            out[f"{col}_te"] = vals.map(self.maps[col]).fillna(self.global_mean).to_numpy()
        return out


def frame_target(y: np.ndarray, dist: np.ndarray, framing: str):
    """Return (y_framed, inverse_fn) for raw / log / per_mile."""
    if framing == "raw":
        return y, (lambda p, d: p)
    if framing == "log":
        return np.log1p(y), (lambda p, d: np.expm1(p))
    if framing == "per_mile":
        return y / np.clip(dist, 1, None), (lambda p, d: p * d)
    raise ValueError(framing)
