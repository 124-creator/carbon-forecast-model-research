"""
har_x_round9.py (v2) — does anything beat HAR out of sample on EU ETS realised variance?

Round 9 diagnostic established the one thing missing elsewhere in this archive: a
target where a model genuinely beats a naive benchmark (HAR +0.3191 OOS R^2 vs
EWMA +0.2117). So the target is contestable and the question is whether extra
information beats HAR itself.

v2 fixes a data defect: the precomputed *_volatility20 columns for Brent, gas and
coal are entirely empty, so v1's energy arm was silently dropped and only
STOXX600/EURUSD survived. Energy features are now built from the price levels
(brent_price, ttf_gas_price), which are populated.

Arms, all nested on identical observations, all strictly as-of t:
  A  HAR                 log RV at 1/5/22 days
  B  HAR + LEV           downside asymmetry
  C  HAR + EN            Brent and TTF gas returns and realised variance
  D  HAR + EN + MAC      plus STOXX600 and EURUSD realised variance
  E  HAR + EN + MAC+LEV  everything

Evaluation: expanding window, refit every 22 observations, test from 2019 and from
2022; targets ly5 (primary), ly1 and ly22. Reported: OOS R^2, MSE, and a block
bootstrap on the squared-error difference against HAR. Only a CI that excludes
zero counts as a win. Twelve cells are examined, so a single significant cell is
not a result.
"""

import numpy as np
import pandas as pd

PX = (r"E:\Agent员工\碳价格预测研究\独立数据归档\_legacy_202605_archive"
      r"\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\01_数据\EUA_原始数据_完整版.csv")

COLS = ["eua_close_t", "brent_price", "ttf_gas_price", "stoxx600_price", "eurusd_price"]


def logrv(s, win):
    r = np.log(s.astype(float)).diff()
    return np.log(r.pow(2).rolling(win).mean().clip(lower=1e-12))


def load():
    d = pd.read_csv(PX, usecols=["date"] + COLS)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)

    d["logp"] = np.log(d["eua_close_t"].astype(float))
    d["r"] = d["logp"].diff()
    d["rv1"] = d["r"] ** 2
    d["rv5"] = d["rv1"].rolling(5).mean()
    d["rv22"] = d["rv1"].rolling(22).mean()
    for h in (1, 5, 22):
        d[f"y{h}"] = d["rv1"].rolling(h).mean().shift(-h)
        d[f"ly{h}"] = np.log(d[f"y{h}"].clip(lower=1e-12))
    d["lrv1"] = np.log(d["rv1"].clip(lower=1e-12))
    d["lrv5"] = np.log(d["rv5"].clip(lower=1e-12))
    d["lrv22"] = np.log(d["rv22"].clip(lower=1e-12))

    # asymmetry / leverage
    d["neg1"] = np.minimum(d["r"], 0.0)
    d["neg5"] = d["neg1"].rolling(5).mean()
    d["semivar5"] = np.log((d["neg1"] ** 2).rolling(5).mean().clip(lower=1e-12))

    # energy and macro, built from price levels
    for tag, col in (("br", "brent_price"), ("gs", "ttf_gas_price"),
                     ("sx", "stoxx600_price"), ("fx", "eurusd_price")):
        s = d[col]
        d[f"{tag}_r"] = np.log(s.astype(float)).diff()
        d[f"{tag}_rv5"] = np.log((d[f"{tag}_r"] ** 2).rolling(5).mean().clip(lower=1e-12))
        d[f"{tag}_rv22"] = np.log((d[f"{tag}_r"] ** 2).rolling(22).mean().clip(lower=1e-12))

    used = [c for c in d.columns if c not in ("date",) + tuple(COLS)]
    print(f"rows={len(d)}  {d['date'].min().date()} -> {d['date'].max().date()}  built={len(used)} features")
    return d


HAR = ["lrv1", "lrv5", "lrv22"]
LEV = ["neg1", "neg5", "semivar5"]
EN = ["br_r", "br_rv5", "br_rv22", "gs_r", "gs_rv5", "gs_rv22"]
MAC = ["sx_rv5", "sx_rv22", "fx_rv5", "fx_rv22"]

ARMS = {
    "A_HAR": HAR,
    "B_HAR_LEV": HAR + LEV,
    "C_HAR_EN": HAR + EN,
    "D_HAR_EN_MAC": HAR + EN + MAC,
    "E_ALL": HAR + EN + MAC + LEV,
}


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b


def walk_forward(d, cols, target, start, refit=22):
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float)
    pred = np.full(len(d), np.nan)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    i = int((d["date"] < start).sum())
    while i < len(d):
        m = ok.copy()
        m[i:] = False
        if m.sum() < 100:
            break
        b = ols(X[m], y[m])
        j = min(i + refit, len(d))
        pred[i:j] = np.column_stack([np.ones(j - i), X[i:j]]) @ b
        i = j
    return pred


def block_ci(diff, block=10, reps=2000, seed=20261021):
    rng = np.random.default_rng(seed)
    n = len(diff)
    nb = int(np.ceil(n / block))
    out = np.empty(reps)
    for r in range(reps):
        st = rng.integers(0, max(1, n - block), nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        out[r] = diff[idx].mean()
    return out


def main():
    d = load()
    for target in ("ly5", "ly22", "ly1"):
        print(f"\n############ target = {target} ############")
        for start in ("2019-01-01", "2022-01-01"):
            truth = d[target].to_numpy(float)
            preds = {n: walk_forward(d, c, target, start) for n, c in ARMS.items()}
            common = np.isfinite(truth) & (d["date"] >= start).to_numpy()
            for p in preds.values():
                common &= np.isfinite(p)
            print(f"  --- test from {start}   n={int(common.sum())} ---")
            if common.sum() < 50:
                print("    too few observations")
                continue
            base = None
            for name in ARMS:
                e2 = (truth[common] - preds[name][common]) ** 2
                r2 = 1.0 - e2.mean() / np.var(truth[common])
                if name == "A_HAR":
                    base = e2
                    print(f"    {name:<13} OOS_R2={r2:+.4f} MSE={e2.mean():.5f}")
                    continue
                diff = e2 - base
                lo, hi = np.percentile(block_ci(diff), [2.5, 97.5])
                star = " *" if hi < 0 else ""
                print(f"    {name:<13} OOS_R2={r2:+.4f} MSE={e2.mean():.5f}"
                      f"  dMSE={diff.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}]{star}")


if __name__ == "__main__":
    main()
