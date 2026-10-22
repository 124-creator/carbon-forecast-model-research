"""
sthar_round9.py — is the EUA volatility process parameter-stable, and does a
state-dependent HAR beat the constant-coefficient HAR out of sample?

Round 9 closed three channels (energy spillovers, auction pressure, conditional
shape) and left one robust effect: downside asymmetry improves 22-day forecasts
and nothing at 1 or 5 days. That horizon-specific pattern is consistent with
parameter instability rather than a missing variable, so this tests instability
directly in two ways.

  1. Model-free: HAR estimated on a rolling 500-day window versus expanding. If a
     short window wins, the coefficients move.
  2. Model: smooth-transition HAR (ST-HAR). The three HAR coefficients are
     multiplied by a logistic transition in a standardised state variable, so the
     persistence of volatility depends on the regime.

State variables, all known at t:
  price  log EUA price — cheap versus expensive carbon
  vol    log 22-day realised variance — calm versus turbulent
  corr   60-day correlation of EUA and Brent returns — decoupled versus coupled

Standardisation uses training data only inside each refit. Walk-forward, refit
every 22 days, targets ly22 (where asymmetry worked), ly5, ly1.
"""

import numpy as np
import pandas as pd

import har_x_round9 as H

REFIT = 22


def build_states(d):
    d = d.copy()
    d["s_price"] = np.log(d["eua_close_t"].astype(float))
    d["s_vol"] = d["lrv22"]
    c = d["r"].rolling(60).corr(d["br_r"])
    d["s_corr"] = c
    return d


def design(X, state, mu, sd, st_mu, st_sd):
    g = 1.0 / (1.0 + np.exp(-np.clip((state - st_mu) / st_sd, -30, 30)))
    return np.column_stack([np.ones(len(X)), X, X * g.reshape(-1, 1), g])


def walk(d, cols, target, start, state=None, window=None):
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float)
    s = d[state].to_numpy(float) if state else np.zeros(len(d))
    ok = np.isfinite(y) & np.isfinite(X).all(1) & (np.isfinite(s) if state else True)
    pred = np.full(len(d), np.nan)
    i = int((d["date"] < start).sum())
    while i < len(d):
        m = ok.copy()
        m[i:] = False
        if window:
            m[: max(0, i - window)] = False
        if m.sum() < 150:
            break
        if state:
            st_mu, st_sd = s[m].mean(), s[m].std() + 1e-9
        else:
            st_mu, st_sd = 0.0, 1.0
        D = design(X[m], s[m], None, None, st_mu, st_sd)
        b, *_ = np.linalg.lstsq(D, y[m], rcond=None)
        j = min(i + REFIT, len(d))
        pred[i:j] = design(X[i:j], s[i:j], None, None, st_mu, st_sd) @ b
        i = j
    return pred


def main():
    d = build_states(H.load())
    arms = {
        "HAR": dict(cols=H.HAR),
        "HAR_roll500": dict(cols=H.HAR, window=500),
        "ST_price": dict(cols=H.HAR, state="s_price"),
        "ST_vol": dict(cols=H.HAR, state="s_vol"),
        "ST_corr": dict(cols=H.HAR, state="s_corr"),
    }
    for target in ("ly22", "ly5", "ly1"):
        print(f"\n############ target = {target} ############")
        for start in ("2019-01-01", "2022-01-01"):
            truth = d[target].to_numpy(float)
            preds = {n: walk(d, target=target, start=start, **kw) for n, kw in arms.items()}
            common = np.isfinite(truth) & (d["date"] >= start).to_numpy()
            for p in preds.values():
                common &= np.isfinite(p)
            print(f"  --- test from {start}   n={int(common.sum())} ---")
            if common.sum() < 50:
                print("    too few"); continue
            base = None
            for name in arms:
                e2 = (truth[common] - preds[name][common]) ** 2
                r2 = 1.0 - e2.mean() / np.var(truth[common])
                if name == "HAR":
                    base = e2
                    print(f"    {name:<12} OOS_R2={r2:+.4f} MSE={e2.mean():.5f}")
                    continue
                diff = e2 - base
                lo, hi = np.percentile(H.block_ci(diff), [2.5, 97.5])
                star = " *" if hi < 0 else ("  (worse)" if lo > 0 else "")
                print(f"    {name:<12} OOS_R2={r2:+.4f} MSE={e2.mean():.5f}"
                      f"  dMSE={diff.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}]{star}")


if __name__ == "__main__":
    main()
