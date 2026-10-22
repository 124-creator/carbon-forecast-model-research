"""Step 1 of the pilot-price replication: data audit + the persistence floor.

Adopts the protocol of Yi (2025, PLoS ONE 20(6):e0326926) exactly where we can:
daily closing price, first 85% of the sample as training and the last 15% as
test, metrics MSE / RMSE / MAE / MAPE, and (unlike that paper) one extra row:
last-value persistence.

The point of the persistence row is not to audit the literature. It is to locate
the floor, so we know what a model has to beat before any "improvement" means
anything.
"""

import os

import numpy as np
import pandas as pd

PANEL = (r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
         r"\research\fullcycle_h20_20261020\PANEL_4MARKET.csv")
START = "2014-04-04"          # Yi (2025) sample start
TRAIN_FRAC = 0.85


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    res = y - yhat
    return {
        "n": int(y.size),
        "MSE": float(np.mean(res ** 2)),
        "RMSE": float(np.sqrt(np.mean(res ** 2))),
        "MAE": float(np.mean(np.abs(res))),
        "MAPE_pct": float(np.mean(np.abs(res / y)) * 100.0),
        "R2": float(1.0 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)),
    }


df = pd.read_csv(PANEL, parse_dates=["date"])
print("raw rows %d  markets %s" % (len(df), sorted(df["market"].unique())))

print("\n--- data audit (before filtering) ---")
print("%-10s %6s %12s %12s %8s %8s %8s" %
      ("market", "n", "first", "last", "zero", "neg", "dupes"))
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    print("%-10s %6d %12s %12s %8d %8d %8d"
          % (mkt, len(g), str(g["date"].min().date()), str(g["date"].max().date()),
             int((g["price"] == 0).sum()), int((g["price"] < 0).sum()),
             int(g["date"].duplicated().sum())))

print("\n--- persistence floor, first 85%% train / last 15%% test, from %s ---" % START)
print("%-10s %6s %12s %14s %14s %12s %10s %10s"
      % ("market", "n_test", "test_start", "MSE", "RMSE", "MAE", "MAPE%", "R2"))
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    g = g[g["date"] >= START].reset_index(drop=True)
    y = g["price"].to_numpy(float)
    if y.size < 100:
        continue
    cut = int(len(y) * TRAIN_FRAC)
    te = np.arange(cut, len(y))
    yhat = np.r_[np.nan, y[:-1]]            # tomorrow = today
    m = metrics(y[te], yhat[te])
    print("%-10s %6d %12s %14.4e %14.4e %12.4f %10.4f %10.6f"
          % (mkt, m["n"], str(g["date"].iloc[cut].date()), m["MSE"], m["RMSE"],
             m["MAE"], m["MAPE_pct"], m["R2"]))

print("\n--- what the training-period mean would give (a constant forecast) ---")
print("%-10s %12s %12s %10s" % ("market", "MAE", "RMSE", "R2"))
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    g = g[g["date"] >= START].reset_index(drop=True)
    y = g["price"].to_numpy(float)
    if y.size < 100:
        continue
    cut = int(len(y) * TRAIN_FRAC)
    yhat = np.full_like(y, np.nan)
    yhat[cut:] = y[:cut].mean()
    m = metrics(y[cut:], yhat[cut:])
    print("%-10s %12.4f %12.4f %10.6f" % (mkt, m["MAE"], m["RMSE"], m["R2"]))
