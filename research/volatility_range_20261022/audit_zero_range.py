"""Volatility audit: is a zero-range day a low-volatility day or a no-trade day?

Every range-based volatility estimator (Parkinson, Garman-Klass, Rogers-Satchell,
Yang-Zhang) is a function of log(H/L). When H == L that quantity is exactly zero,
so the estimator reports zero variance. If those days are genuinely quiet days the
treatment is right. If they are days on which the instrument did not trade, the
estimator is recording a missing observation as a realised value of zero, which
biases the volatility series down and distorts every model built on it.

Volume separates the two hypotheses without any modelling assumption:
  quiet day    -> small but positive volume
  no-trade day -> zero volume (or the file carries no trade at all)

This script only audits. No forecasting yet.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SRC = (r"E:\Agent员工\碳价格预测研究\独立数据归档\_legacy_202605_archive"
       r"\主要实验推进_审计加固版_CLEAN_20260528\08-新实验"
       r"\22_structural_ohlc_q2_route_20260609\03_raw_data\normalized_yahoo_ohlc")
OUT = os.path.dirname(os.path.abspath(__file__))
SYMS = ("KRBN", "KEUA", "KCCA", "CL=F", "NG=F")

rows = []
for s in SYMS:
    p = os.path.join(SRC, "%s_ohlc.csv" % s)
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d["high"] >= d["low"]) & (d["close"] > 0)]
    n = len(d)
    if n < 250:
        continue
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = d["volume"].to_numpy(float)

    flat = np.isclose(h, l, rtol=0, atol=1e-10)
    dead = flat & np.isclose(o, c, rtol=0, atol=1e-10)
    novol = v <= 0

    both = flat & novol
    flat_only = flat & ~novol

    # share of total variance that disappears when flat days are scored as zero
    lr = np.log(h / l)
    park = lr ** 2 / (4 * np.log(2))
    share_flat_var = float(park[flat].sum() / park.sum()) if park.sum() > 0 else np.nan

    # what a flat day's neighbours look like: if flat days sit inside normal
    # volatility, they are missing data rather than genuinely quiet days
    nb = np.r_[park[1:], np.nan]
    nb_prev = np.r_[np.nan, park[:-1]]
    neigh = np.nanmean(np.c_[nb, nb_prev], axis=1)
    ratio = np.nanmedian(neigh[flat]) / np.nanmedian(neigh[~flat]) if flat.any() else np.nan

    rows.append(dict(symbol=s, n=n, first=str(d["date"].iloc[0].date()),
                     last=str(d["date"].iloc[-1].date()),
                     pct_flat=100 * flat.mean(),
                     pct_flat_dead=100 * dead.mean(),
                     pct_novol=100 * novol.mean(),
                     pct_flat_and_novol=100 * both.mean(),
                     pct_flat_but_vol=100 * flat_only.mean(),
                     share_var_lost=100 * share_flat_var,
                     neigh_ratio=ratio,
                     med_vol=float(np.median(v))))

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("--- zero-range audit ---")
print(res.to_string(index=False, float_format=lambda x: "%.2f" % x))

print("\n--- interpretation ---")
for _, r in res.iterrows():
    verdict = ("NO-TRADE (flat days carry no volume)"
               if r["pct_flat_but_vol"] < 0.25 * max(r["pct_flat"], 1e-9)
               else "MIXED (flat days still trade)")
    print("%-6s flat %5.1f%%  of which zero-volume %5.1f%%  -> %s"
          % (r["symbol"], r["pct_flat"], 100 - r["pct_flat_but_vol"], verdict))

print("\n--- consecutive flat runs (how long does a stale spell last?) ---")
for s in SYMS:
    p = os.path.join(SRC, "%s_ohlc.csv" % s)
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d["high"] >= d["low"]) & (d["close"] > 0)]
    if len(d) < 250:
        continue
    flat = np.isclose(d["high"].to_numpy(float), d["low"].to_numpy(float),
                      rtol=0, atol=1e-10).astype(int)
    runs, cur = [], 0
    for f in flat:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    runs = np.asarray(runs) if runs else np.array([0])
    print("%-6s runs=%4d  max=%3d  mean=%.2f  share of flat days in runs>=2: %.1f%%"
          % (s, runs.size, runs.max(), runs.mean(),
             100 * runs[runs >= 2].sum() / max(runs.sum(), 1)))

res.to_csv(os.path.join(OUT, "AUDIT_ZERO_RANGE.csv"), index=False, encoding="utf-8-sig")
print("\nwrote AUDIT_ZERO_RANGE.csv")
