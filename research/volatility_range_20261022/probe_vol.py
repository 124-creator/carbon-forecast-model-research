"""Feasibility probe, corrected: HAR vs HAR-X with foreign carbon volatility.

Q1 (persistence) and Q2 (cross-market) already answered in the first run:
log-RV autocorrelation is 0.32/0.33/0.51 at lag one, and same-day log-RV
correlation across carbon instruments is 0.17-0.49 -- one to two orders of
magnitude larger than the 0.004-0.065 we measured for the price LEVEL. That
asymmetry is the candidate story.

This run builds the design matrices once, correctly, and asks whether the foreign
volatility term survives out of sample in QLIKE.
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
SYMS = ("KRBN", "KEUA", "KCCA")
LOG2 = np.log(2.0)
HAR_LAGS = (1, 5, 22)


def load(s):
    d = pd.read_csv(os.path.join(SRC, "%s_ohlc.csv" % s), parse_dates=["date"])
    d = d.dropna(subset=["open", "high", "low", "close"]).sort_values("date")
    d = d[(d["high"] >= d["low"]) & (d["close"] > 0)]
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    lr = np.log(np.maximum(h, l * (1 + 1e-12)) / l)
    park = lr ** 2 / (4 * LOG2)
    ret = np.r_[np.nan, np.diff(np.log(c))]
    rv = np.where(park > 1e-12, park, ret ** 2)
    rv = np.where(np.isfinite(rv) & (rv > 0), rv, np.nan)
    return pd.Series(rv, index=d["date"], name=s)


panel = pd.concat([load(s) for s in SYMS], axis=1).sort_index()
panel = panel.dropna(how="all")


def har_cols(x, i):
    """Mean of x over the last L observations strictly before i."""
    out = []
    for L in HAR_LAGS:
        seg = x[max(0, i - L):i]
        out.append(np.nanmean(seg) if seg.size else np.nan)
    return out


def build(target, drivers):
    y = panel[target].to_numpy(float)
    n = y.size
    ly = np.log(y)
    LD = [np.log(panel[d].to_numpy(float)) for d in drivers]
    rows, Yt, Xown, Xfgn = [], [], [], []
    for i in range(60, n - 1):
        own = har_cols(ly, i)
        if drivers:
            fgn = np.concatenate([har_cols(LD[k], i) for k in range(len(drivers))])
        else:
            fgn = np.zeros(0)
        rows.append(i)
        Yt.append(ly[i + 1])
        Xown.append(own)
        Xfgn.append(fgn)
    return (y, np.asarray(rows), np.asarray(Yt),
            np.asarray(Xown, float), np.asarray(Xfgn, float))


def qlike(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return float(np.mean(r - np.log(r) - 1.0))


def run(target, drivers):
    y, rows, Y, Xo, Xf = build(target, drivers)
    n = y.size
    t0 = int(n * 0.50)
    starts = np.searchsorted(rows, t0)
    preds = {k: [] for k in ("RW", "MA22", "HAR", "HARX")}
    act = []
    for j in range(starts, len(rows) - 1):
        i = rows[j]
        tr = np.arange(0, j)                       # targets known by origin i
        ok = np.isfinite(Y[tr]) & np.isfinite(Xo[tr]).all(1)
        tr = tr[ok]
        if tr.size < 150:
            continue
        A = np.c_[np.ones(tr.size), Xo[tr]]
        b = np.linalg.lstsq(A, Y[tr], rcond=None)[0]
        if not np.isfinite(Xo[j]).all():
            continue
        f_har = float(np.exp(np.r_[1.0, Xo[j]] @ b))

        okx = ok.copy()
        okx[tr] = np.isfinite(Xf[tr]).all(1)
        trx = np.arange(0, j)[okx]
        if trx.size > 200 and np.isfinite(Xf[j]).all():
            Ax = np.c_[A[np.isin(tr, trx)], Xf[trx]]
            bx = np.linalg.lstsq(Ax, Y[trx], rcond=None)[0]
            f_x = float(np.exp(np.r_[1.0, Xo[j], Xf[j]] @ bx))
        else:
            f_x = f_har

        preds["HAR"].append(f_har)
        preds["HARX"].append(f_x)
        preds["RW"].append(float(y[i]))
        preds["MA22"].append(float(np.nanmean(y[max(0, i - 22):i + 1])))
        act.append(float(y[i + 1]))

    a = np.asarray(act, float)
    out = {}
    for k, v in preds.items():
        f = np.asarray(v, float)
        ok = np.isfinite(f) & (f > 0) & np.isfinite(a)
        out[k] = qlike(a[ok], f[ok])
    base = out["HAR"]
    print("\n%s  vs %s   (n_test=%d)" % (target, "+".join(drivers), a.size))
    for k in ("RW", "MA22", "HAR", "HARX"):
        print("   %-5s QLIKE %+.6f   %+7.2f%% vs HAR" % (k, out[k],
                                                         (out[k] / base - 1) * 100))
    return out


print("=== Q3: does foreign carbon volatility improve HAR out of sample? ===")
r1 = run("KRBN", ["KEUA", "KCCA"])
r2 = run("KEUA", ["KRBN", "KCCA"])
r3 = run("KCCA", ["KRBN", "KEUA"])

print("\n=== own-history only (is HAR even better than random walk?) ===")
for s in SYMS:
    o = run(s, [])
