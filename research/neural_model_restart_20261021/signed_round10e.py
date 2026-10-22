"""
signed_round10e.py — paper-grade validation of the one-parameter extension.

The model that survived every control is

    HAR-D:  lrv1 lrv5 lrv22  +  g22,   g22 = log downside variance(22) + log upside variance(22)

which is one extra parameter. Because lrv22 is the log of the arithmetic mean of the
two signed components and g22 is twice the log of their geometric mean, the pair
carries the level plus AM/GM, a one-sidedness index: how directional the month's
variation was, independent of its size.

This script adds the tests a referee will ask for and which the earlier rounds did
not provide:

  1. Diebold-Mariano with a HAC variance, on the QLIKE loss differential.
  2. The same model scored on log variance with MSE and R^2, so the result does not
     rest on one loss function.
  3. Mincer-Zarnowitz calibration: regress the outcome on the forecast, test slope 1
     and intercept 0.
  4. Forecast encompassing: does HAR add anything once HAR-D is known, and the
     reverse. A one-parameter extension should encompass the nested model.
  5. The five markets at the monthly horizon, using the one-parameter specification.
"""

import numpy as np
import pandas as pd

import har_x_round9 as H

REFIT = 22
DM_LAG = 44
PANEL = (r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
         r"\research\fullcycle_h20_20261020\PANEL_5MARKET.csv")
MIN_TRAIN = 500


def add_g(d):
    d = d.copy()
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    d["dn22"] = np.log(neg.rolling(22).mean().clip(lower=1e-14))
    d["up22"] = np.log(pos.rolling(22).mean().clip(lower=1e-14))
    d["g22"] = d["dn22"] + d["up22"]
    return d


def walk(d, cols, target, start=None, refit=REFIT):
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float)
    v = np.full(len(d), np.nan)
    p = np.full(len(d), np.nan)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    if start is None:
        cum = np.cumsum(ok)
        if cum[-1] < MIN_TRAIN + 200:
            return v, p
        i = int(np.argmax(cum >= MIN_TRAIN)) + 1
    else:
        i = int((d["date"] < start).sum())
    while i < len(d):
        m = ok.copy()
        m[i:] = False
        if m.sum() < 150:
            break
        D = np.column_stack([np.ones(m.sum()), X[m]])
        b, *_ = np.linalg.lstsq(D, y[m], rcond=None)
        f = float(np.mean(np.exp(y[m] - D @ b)))
        j = min(i + refit, len(d))
        p[i:j] = np.column_stack([np.ones(j - i), X[i:j]]) @ b
        v[i:j] = np.exp(p[i:j]) * f
        i = j
    return v, p


def ql(v, y):
    r = np.clip(v, 1e-14, None) / np.clip(y, 1e-14, None)
    return r - np.log(r) - 1.0


def dm(diff, lag=DM_LAG):
    n = len(diff)
    mu = diff.mean()
    e = diff - mu
    s = float(e @ e) / n
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        s += 2.0 * w * float(e[L:] @ e[:-L]) / n
    if s <= 0:
        return np.nan, np.nan
    stat = mu / np.sqrt(s / n)
    from math import erfc, sqrt
    return stat, erfc(abs(stat) / sqrt(2.0))


def mz(y, f):
    D = np.column_stack([np.ones(len(f)), f])
    b, *_ = np.linalg.lstsq(D, y, rcond=None)
    e = y - D @ b
    s2 = float(e @ e) / (len(f) - 2)
    V = s2 * np.linalg.pinv(D.T @ D)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return b, se


def main():
    d = add_g(H.load())
    print("=" * 78)
    print("EUA, one-parameter extension HAR-D = HAR + g22")
    print("=" * 78)
    for target, proxy in (("ly5", "y5"), ("ly22", "y22")):
        for start in ("2019-01-01", "2022-01-01"):
            v0, p0 = walk(d, H.HAR, target, start)
            v1, p1 = walk(d, H.HAR + ["g22"], target, start)
            m = np.isfinite(d[proxy]).to_numpy() & (d["date"] >= start).to_numpy()
            m &= np.isfinite(v0) & np.isfinite(v1) & np.isfinite(p0) & np.isfinite(p1)
            y = d[proxy].to_numpy()[m]
            l0 = ql(v0[m], y)
            l1 = ql(v1[m], y)
            stat, pv = dm(l1 - l0)
            tgt = d[target].to_numpy()[m]
            mse0 = float(np.mean((p0[m] - tgt) ** 2))
            mse1 = float(np.mean((p1[m] - tgt) ** 2))
            dm2, pv2 = dm((p1[m] - tgt) ** 2 - (p0[m] - tgt) ** 2)
            print(f"\n  {target} from {start}   n={int(m.sum())}")
            print(f"    QLIKE   HAR={l0.mean():.5f}  HAR-D={l1.mean():.5f}  "
                  f"gain={100*(1-l1.mean()/l0.mean()):+.2f}%  DM={stat:+.2f} p={pv:.4f}")
            print(f"    MSE_log HAR={mse0:.5f}  HAR-D={mse1:.5f}  "
                  f"gain={100*(1-mse1/mse0):+.2f}%  DM={dm2:+.2f} p={pv2:.4f}")
            b, se = mz(tgt, p1[m])
            print(f"    Mincer-Zarnowitz on HAR-D: a={b[0]:+.4f} (se {se[0]:.4f})  "
                  f"b={b[1]:+.4f} (se {se[1]:.4f})")
            # encompassing, in logs
            D = np.column_stack([np.ones(int(m.sum())), p0[m], p1[m]])
            be, *_ = np.linalg.lstsq(D, tgt, rcond=None)
            print(f"    encompassing: target on (HAR, HAR-D) -> "
                  f"coef HAR={be[1]:+.3f}  HAR-D={be[2]:+.3f}")

    print("\n" + "=" * 78)
    print("five markets, monthly horizon, one-parameter extension")
    print("=" * 78)
    raw = pd.read_csv(PANEL)
    raw["date"] = pd.to_datetime(raw["date"])
    for mkt in ["EU", "CEA", "GUANGZHOU", "SHANGHAI", "HUBEI"]:
        g = raw[raw["market"] == mkt][["date", "price"]].dropna()
        g = g.sort_values("date").reset_index(drop=True)
        g["r"] = np.log(g["price"].astype(float)).diff()
        r2 = g["r"] ** 2
        for w, c in ((1, "rv1"), (5, "rv5"), (22, "rv22")):
            g[c] = r2.rolling(w).mean()
            g["l" + c] = np.log(g[c].clip(lower=1e-14))
        g["y22"] = r2.rolling(22).mean().shift(-22)
        g["ly22"] = np.log(g["y22"].clip(lower=1e-14))
        g = add_g(g)
        v0, _ = walk(g, H.HAR, "ly22")
        v1, _ = walk(g, H.HAR + ["g22"], "ly22")
        m = (np.isfinite(g["y22"]).to_numpy() & np.isfinite(v0) & np.isfinite(v1))
        if m.sum() < 200:
            print(f"  {mkt:<10} unusable (n={int(m.sum())})")
            continue
        y = g["y22"].to_numpy()[m]
        a, b = ql(v0[m], y), ql(v1[m], y)
        stat, pv = dm(b - a, lag=44)
        print(f"  {mkt:<10} n={int(m.sum()):>4}  HAR={a.mean():.4f}  "
              f"HAR-D={b.mean():.4f}  gain={100*(1-b.mean()/a.mean()):+6.2f}%  "
              f"DM={stat:+.2f} p={pv:.3f}")


if __name__ == "__main__":
    main()
