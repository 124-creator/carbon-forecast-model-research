"""Step 8: is the reversion gain real, or an artifact of one 85/15 window?

Everything so far is a single split. Step 4's Hubei h=22 result (-5.68% MAE,
-12.77% RMSE vs persistence) is the only sizeable positive signal found, and a
single 200-point window can produce that by luck.

So: rolling-origin evaluation. At every origin in the later part of each series,
fit AR(p) on the data available AT THAT MOMENT (expanding, no look-ahead),
forecast h steps, and score against the realised value and against persistence.
That yields thousands of paired comparisons per market instead of ~200, and a
Diebold-Mariano style test with a Newey-West correction for the overlap induced
by multi-step horizons.

If the gain does not survive this, it was a window artifact and there is no
price-history model here to build on.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PANEL = (r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
         r"\research\fullcycle_h20_20261020\PANEL_4MARKET.csv")
START = "2014-04-04"
ORIGIN_FRAC = 0.60          # origins start after 60% of the sample
HORIZONS = (1, 5, 22)


def ar_fit(z, p):
    X, t = [], []
    for i in range(p, len(z)):
        X.append(z[i - p:i][::-1])
        t.append(z[i])
    X, t = np.asarray(X), np.asarray(t)
    A = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(A, t, rcond=None)[0]


def ar_fc(z, beta, h):
    p = len(beta) - 1
    buf = list(z[-p:])
    for _ in range(h):
        buf.append(beta[0] + np.dot(beta[1:], np.asarray(buf[-p:])[::-1]))
    return buf[-1]


def nw_t(d, lag):
    """Mean of paired loss differential d, with a Newey-West t-statistic."""
    d = np.asarray(d, float)
    n = d.size
    m = d.mean()
    e = d - m
    v = np.dot(e, e) / n
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        v += 2.0 * w * np.dot(e[L:], e[:-L]) / n
    se = np.sqrt(v / n)
    return m, se, (m / se if se > 0 else np.nan)


df = pd.read_csv(PANEL, parse_dates=["date"])
rows = []
detail = {}
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    g = g[g["date"] >= START].reset_index(drop=True)
    y = g["price"].to_numpy(float)
    if y.size < 300:
        continue
    z = np.log(y)
    t0 = int(len(z) * ORIGIN_FRAC)

    for p in (1, 2):
        for h in HORIZONS:
            ae_ar, ae_p = [], []
            for t in range(t0, len(z) - h):
                beta = ar_fit(z[:t + 1], p)
                yh = np.exp(ar_fc(z[:t + 1], beta, h))
                ae_ar.append(abs(y[t + h] - yh))
                ae_p.append(abs(y[t + h] - y[t]))
            ae_ar, ae_p = np.asarray(ae_ar), np.asarray(ae_p)
            d = ae_ar - ae_p
            m, se, tt = nw_t(d, lag=max(h - 1, 0))
            rows.append(dict(market=mkt, order=p, h=h, n=d.size,
                             MAE_ar=ae_ar.mean(), MAE_pers=ae_p.mean(),
                             dMAE=(ae_ar.mean() / ae_p.mean() - 1) * 100,
                             diff=m, se=se, t=tt))
            if p == 2 and h == 22:
                detail[mkt] = (ae_ar, ae_p)
    print("done %s (origins from index %d of %d)" % (mkt, t0, len(z)))

res = pd.DataFrame(rows)
print("\n--- rolling-origin, expanding fit, no look-ahead ---")
print("%-10s %2s %3s %5s %10s %10s %9s %11s %11s %8s"
      % ("market", "p", "h", "n", "MAE_AR", "MAE_pers", "dMAE%", "diff", "se", "NW_t"))
for mkt in res["market"].unique():
    for _, r in res[res["market"] == mkt].sort_values(["h", "order"]).iterrows():
        print("%-10s %2d %3d %5d %10.4f %10.4f %+9.2f %+11.5f %11.5f %+8.2f"
              % (mkt, r["order"], r["h"], r["n"], r["MAE_ar"], r["MAE_pers"],
                 r["dMAE"], r["diff"], r["se"], r["t"]))
    print()

print("--- verdict per (h, p): does reversion beat persistence? ---")
for h in HORIZONS:
    for p in (1, 2):
        s = res[(res["h"] == h) & (res["order"] == p)]
        wins = int((s["dMAE"] < 0).sum())
        sig = int(((s["dMAE"] < 0) & (s["t"] < -2)).sum())
        print("  h=%-2d p=%d : beats persistence on %d/%d markets, significant on %d"
              % (h, p, wins, len(s), sig))

base = os.path.dirname(os.path.abspath(__file__))
res.to_csv(os.path.join(base, "STEP8_ROLLING_ORIGIN.csv"), index=False, encoding="utf-8-sig")
np.savez(os.path.join(base, "STEP8_detail.npz"),
         **{("ae_ar_%s" % k): v[0] for k, v in detail.items()},
         **{("ae_p_%s" % k): v[1] for k, v in detail.items()})
print("\nwrote STEP8_ROLLING_ORIGIN.csv and STEP8_detail.npz")
