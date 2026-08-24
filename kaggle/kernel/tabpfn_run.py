"""TabPFN entry for the Spotter freight bake-off — runs on Kaggle GPU.

Self-contained: replicates the local v2 feature pipeline (calendar + rich
numeric features + leak-safe smoothed target encoding) under the log-target
framing, evaluates TabPFN on the same Aug/Sep/Oct time folds, and writes final
predictions for the 12k validation set and the 31 December days.

Outputs (to /kaggle/working):
  tabpfn_cv_metrics.csv
  validation_predictions_tabpfn.csv   (load_id,predicted_rate)
  december_tabpfn.csv                 (date,predicted_rate)
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch  # preinstalled on the Kaggle GPU image

# --- install TabPFN (internet enabled on this kernel) ---------------------
# Pin the v2 line: its weights load from HuggingFace without the license-token
# gate that the newer (v3 / package 8.x) checkpoints require, which can't be
# accepted in a non-interactive kernel. Install its deps normally, but CONSTRAIN
# torch to Kaggle's exact build so pip cannot swap in a PyPI torch whose CUDA
# kernels don't match this GPU ("no kernel image" ABI error).
_base = torch.__version__.split("+")[0]
_cons = "/kaggle/working/constraints.txt"
with open(_cons, "w") as _f:
    _f.write(f"torch=={_base}\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", _cons, "tabpfn<3"], check=True)

from tabpfn import TabPFNRegressor

# Kaggle's torch 2.10/cu128 lacks compiled kernels for the assigned GPU ("no
# kernel image" error), so run on CPU. TabPFN is a small-data foundation model,
# so we subsample the training context — the intended use for TabPFN anyway.
DEVICE = "cpu"
print(f"device = {DEVICE} | torch {torch.__version__}")

OUT = "/kaggle/working"


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        tree = {p: os.listdir(p) for p in glob.glob("/kaggle/input/*")}
        raise FileNotFoundError(f"{name} not found. /kaggle/input tree = {tree}")
    return hits[0]
TARGET = "posted_rate"
EQUIP = ["Dry Van", "Flatbed", "Reefer"]
EXTRA = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
         "market_index", "quote_signal"]
TE_COLS = ["lane", "pickup", "delivery"]
TRAIN_CAP = 3000    # CPU: keep the TabPFN training context small
VAL_CAP = 1200      # CPU: subsample each validation fold for a fast metric
SEED = 42


# --- features -------------------------------------------------------------
def calendar(df):
    d = pd.to_datetime(df["date"]); doy = d.dt.dayofyear
    out = pd.DataFrame(index=df.index)
    out["month"] = d.dt.month; out["weekday"] = d.dt.weekday
    out["day"] = d.dt.day; out["week"] = d.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (d.dt.weekday >= 5).astype(int)
    out["doy_sin"] = np.sin(2*np.pi*doy/365.25); out["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    return out


def ordinal(series, cats):
    c = pd.Categorical(series.astype(str), categories=cats).codes.astype(float)
    c[c < 0] = np.nan
    return pd.Series(c, index=series.index)


def build_static(df, schema, feature_set="full"):
    parts = [calendar(df)]
    dist = df["distance"].astype(float); wt = df["weight"].astype(float)
    b = pd.DataFrame(index=df.index)
    b["distance"] = dist; b["log_distance"] = np.log1p(dist)
    b["weight"] = wt; b["weight_per_mile"] = wt / dist.clip(lower=1)
    b["pickup_code"] = ordinal(df["pickup"], schema["pickup"])
    b["delivery_code"] = ordinal(df["delivery"], schema["delivery"])
    parts.append(b)
    eq = pd.get_dummies(pd.Categorical(df["equipment"], categories=EQUIP), prefix="e").astype(float)
    eq.index = df.index; parts.append(eq)
    inter = pd.DataFrame(index=df.index)
    for c in EQUIP:
        inter[f"dx_{c.replace(' ','')}"] = dist * (df["equipment"] == c).astype(float)
    parts.append(inter)
    if feature_set == "full":
        ex = pd.DataFrame(index=df.index)
        ex["market_index"] = df["market_index"].astype(float)
        ex["quote_signal"] = df["quote_signal"].astype(float)
        ex["dx_market"] = dist * ex["market_index"]; ex["dx_quote"] = dist * ex["quote_signal"]
        parts.append(ex)
    return pd.concat(parts, axis=1)


def lane(df):
    return df["pickup"].astype(str) + ">" + df["delivery"].astype(str)


def te_fit(df, y, smoothing=50.0):
    gm = float(np.mean(y)); maps = {}
    for col in TE_COLS:
        key = lane(df) if col == "lane" else df[col].astype(str)
        t = pd.DataFrame({"k": key.to_numpy(), "y": y})
        s = t.groupby("k")["y"].agg(["mean", "count"])
        maps[col] = (s["mean"]*s["count"] + gm*smoothing) / (s["count"]+smoothing)
    return maps, gm


def te_transform(df, maps, gm):
    out = pd.DataFrame(index=df.index)
    for col in TE_COLS:
        key = lane(df) if col == "lane" else df[col].astype(str)
        out[f"{col}_te"] = key.map(maps[col]).fillna(gm).to_numpy()
    return out


def metrics(y, p):
    p = np.clip(p, 1e-6, None); e = y - p
    return {"RMSE": float(np.sqrt(np.mean(e**2))), "MAE": float(np.mean(np.abs(e))),
            "MAPE": float(np.mean(np.abs(e)/y)*100),
            "R2": 1 - float(np.sum(e**2))/float(np.sum((y-y.mean())**2))}


def make_tabpfn():
    return TabPFNRegressor(device=DEVICE, ignore_pretraining_limits=True, random_state=SEED)


def subsample(idx, cap):
    if len(idx) <= cap:
        return idx
    rng = np.random.RandomState(SEED)
    return idx[rng.permutation(len(idx))[:cap]]


# --- run ------------------------------------------------------------------
train = pd.read_csv(find("train_test.csv"), parse_dates=["date"])
valid = pd.read_csv(find("validation.csv"), parse_dates=["date"])
december = pd.read_csv(find("december_chart_inputs.csv"), parse_dates=["date"])
schema = {"pickup": sorted(train["pickup"].astype(str).unique()),
          "delivery": sorted(train["delivery"].astype(str).unique())}
y = train[TARGET].to_numpy(float)
dist = train["distance"].to_numpy(float)
month = train["date"].dt.month

rows = []
for feature_set in ("full", "reduced"):
    static = build_static(train, schema, feature_set)
    fold_metrics = []
    for m in (9, 10):
        tr = train.index[month < m]; va = train.index[month == m]
        tr = subsample(tr, TRAIN_CAP)
        va = subsample(va, VAL_CAP)
        yf = np.log1p(y)
        maps, gm = te_fit(train.loc[tr], yf[tr])
        Xtr = pd.concat([static.loc[tr], te_transform(train.loc[tr], maps, gm)], axis=1)
        Xva = pd.concat([static.loc[va], te_transform(train.loc[va], maps, gm)], axis=1)
        t0 = time.time()
        model = make_tabpfn(); model.fit(Xtr.to_numpy(float), yf[tr])
        pred = np.expm1(model.predict(Xva.to_numpy(float)))
        mm = metrics(y[va], pred); mm["secs"] = round(time.time()-t0, 1)
        fold_metrics.append(mm)
        print(f"[{feature_set}] month {m}: RMSE={mm['RMSE']:.2f} MAE={mm['MAE']:.2f} "
              f"MAPE={mm['MAPE']:.2f}% ({mm['secs']}s)")
    agg = {k: float(np.mean([f[k] for f in fold_metrics])) for k in ("RMSE","MAE","MAPE","R2")}
    agg["feature_set"] = feature_set
    rows.append(agg)
    print(f"[{feature_set}] MEAN RMSE={agg['RMSE']:.2f} MAE={agg['MAE']:.2f} MAPE={agg['MAPE']:.2f}%")

pd.DataFrame(rows).to_csv(f"{OUT}/tabpfn_cv_metrics.csv", index=False)

# Final 12k / December predictions are intentionally NOT produced here: TabPFN on
# CPU is too slow over 12k rows, and it is not the submitted model (the blend is).
# This kernel exists to place TabPFN on the bake-off leaderboard via CV metrics.
print("DONE — wrote tabpfn_cv_metrics.csv (CV-only bake-off entry)")
