"""
vol_diag_round9.py — feasibility diagnostic for a volatility target.

Why switch targets: the price target is hopeless because persistence R^2 is
~0.99, leaving nothing for a model to add. Realised volatility is far less
persistent, so forecasting skill is possible in principle. Before building
anything, this checks four things:

  1. how persistent log RV actually is here;
  2. whether a plain HAR beats EWMA out of sample (if HAR has no headroom over a
     trivial benchmark, the target is not worth modelling);
  3. whether auction days differ from non-auction days (the one information set
     that is genuinely ours: 1,281 exact auction dates);
  4. whether auction timing carries information beyond the HAR components.

If (2) fails there is no competition to win. If (3) and (4) are both flat there is
no new information set either. Either failure ends the line.
"""

import numpy as np
import pandas as pd

ROOT = r"E:\Agent员工\碳价格预测研究\独立数据归档"
PX = ROOT + r"\_legacy_202605_archive\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\01_数据\EUA_原始数据_完整版.csv"
AUC = ROOT + r"\EUA\pro_response_bridge_20261011\private_auction_base_reviewed\auction_prices_local.csv"


def load():
    px = pd.read_csv(PX, usecols=["date", "eua_close_t"])
    px["date"] = pd.to_datetime(px["date"])
    px = px.dropna().sort_values("date").reset_index(drop=True)
    px["logp"] = np.log(px["eua_close_t"].astype(float))
    px["r"] = px["logp"].diff()
    # realised variance components (decimal returns, daily units)
    px["rv1"] = px["r"] ** 2
    px["rv5"] = px["rv1"].rolling(5).mean()
    px["rv22"] = px["rv1"].rolling(22).mean()
    # target: mean variance over the NEXT 5 trading days (strictly forward)
    px["y5"] = px["rv1"].rolling(5).mean().shift(-5)
    px["ly5"] = np.log(px["y5"].clip(lower=1e-12))
    px["lrv1"] = np.log(px["rv1"].clip(lower=1e-12))
    px["lrv5"] = np.log(px["rv5"].clip(lower=1e-12))
    px["lrv22"] = np.log(px["rv22"].clip(lower=1e-12))

    a = pd.read_csv(AUC)
    a["date"] = pd.to_datetime(a["auction_date"])
    dates = set(a["date"].dt.normalize())
    px["is_auc"] = px["date"].dt.normalize().isin(dates).astype(float)
    # distance in trading days to the nearest auction (signed, and absolute)
    idx = px.index[px["is_auc"] > 0]
    if len(idx):
        pos = np.arange(len(px))
        near = np.array([pos[np.argmin(np.abs(pos - i))] for i in range(len(px))])
        px["d_auc"] = pos - near
    else:
        px["d_auc"] = np.nan
    px["auc_week"] = (px["d_auc"].abs() <= 2).astype(float)
    px = px.dropna(subset=["ly5", "lrv22"]).reset_index(drop=True)
    return px


def har_fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b


def main():
    d = load()
    print(f"sample {d['date'].min().date()} -> {d['date'].max().date()}  n={len(d)}")
    print(f"auction days in sample: {int(d['is_auc'].sum())}")

    print("\n--- 1. persistence of the target ---")
    for c in ("lrv1", "lrv5", "lrv22", "ly5"):
        s = d[c].dropna()
        print(f"  {c:<7} acf1={s.autocorr(1):.3f}  acf5={s.autocorr(5):.3f}  sd={s.std():.3f}")

    print("\n--- 2. out-of-sample: HAR vs EWMA vs random walk ---")
    tr = d["date"] < "2022-01-01"
    te = d["date"] >= "2022-01-01"
    X = d[["lrv1", "lrv5", "lrv22"]].to_numpy(float)
    y = d["ly5"].to_numpy(float)
    # HAR (expanding, refit every 60 obs)
    pred = np.full(len(d), np.nan)
    for i in range(int(tr.sum()), len(d), 60):
        m = np.zeros(len(d), bool); m[:i] = True
        b = har_fit(X[m], y[m])
        j = min(i + 60, len(d))
        pred[i:j] = b[0] + X[i:j] @ b[1:]
    ewma = d["lrv22"].to_numpy(float)                     # EWMA / persistence proxy
    rw = d["ly5"].shift(5).to_numpy(float)                # naive: recent realised
    truth = d["ly5"].to_numpy(float)
    m = te.to_numpy() & np.isfinite(pred) & np.isfinite(rw)
    print(f"  n_test={int(m.sum())}")
    for name, p in (("HAR", pred), ("EWMA22", ewma), ("RW5", rw)):
        e = truth[m] - p[m]
        r2 = 1.0 - np.mean(e ** 2) / np.var(truth[m])
        print(f"  {name:<7} OOS R2={r2:+.4f}   MSE={np.mean(e**2):.5f}")

    print("\n--- 3. do auction days differ? (same-day) ---")
    sub = d[d["date"] >= "2020-01-01"]
    print(f"  2020+ n={len(sub)}  auction days={int(sub['is_auc'].sum())}")
    g = sub.groupby("is_auc")["lrv1"].agg(["size", "mean", "median", "std"])
    print(g.round(3).to_string())
    # also realised |r| on auction day vs matched non-auction days
    print(f"  mean |r| auction={sub.loc[sub['is_auc']>0,'r'].abs().mean()*1e4:.1f}bp  "
          f"non-auction={sub.loc[sub['is_auc']==0,'r'].abs().mean()*1e4:.1f}bp")

    print("\n--- 4. auction timing beyond HAR components ---")
    Xa = np.column_stack([np.ones(len(sub)), sub[["lrv1", "lrv5", "lrv22"]].to_numpy(float)])
    ya = np.log(sub["y5"].clip(lower=1e-12)).to_numpy(float)
    b0 = har_fit(Xa[:, 1:], ya)
    r0 = ya - (Xa @ b0)
    for col in ("is_auc", "auc_week"):
        Z = np.column_stack([Xa, sub[col].to_numpy(float)])
        b1 = har_fit(Z[:, 1:], ya)
        r1 = ya - (Z @ b1)
        n, k = len(ya), Z.shape[1]
        tstat = b1[-1] / np.sqrt(np.sum(r1 ** 2) / (n - k) * np.linalg.inv(Z.T @ Z)[-1, -1])
        print(f"  {col:<9} coef={b1[-1]:+.4f}  t={tstat:+.2f}  "
              f"R2 {1-np.sum(r0**2)/np.sum((ya-ya.mean())**2):.4f} -> {1-np.sum(r1**2)/np.sum((ya-ya.mean())**2):.4f}")


if __name__ == "__main__":
    main()
