"""
qhar_round9b.py — the decisive test: is the volatility DISTRIBUTION conditionally
state-dependent, or is it just a location shift?

Round 9 showed quantile HAR beats climatology by 15-30% MQS. That is obvious and
unpublishable — of course conditional information beats an unconditional
distribution. The question that decides whether there is a contribution is
whether the *shape* moves with the state.

Two ways to build a predictive distribution from the same information:

  LOC   location-shift: OLS on the conditional mean, then add the training
        residual quantiles (an unconditional shape). This is the standard
        semiparametric benchmark in the volatility literature.
  QNT   direct quantile regression, where every quantile has its own coefficients
        and the shape is therefore allowed to change with the state.

If QNT materially beats LOC, the shape is state-dependent and that is the finding.
If it does not, the distribution is a pure location shift and there is no paper.

Arms (identical observations, walk-forward, refit every 66 days):
  L1  OLS(HAR)        + residual quantiles
  L2  OLS(HAR+LEV)    + residual quantiles
  Q1  quantile HAR
  Q2  quantile HAR+LEV
"""

import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\neural_model_restart_20261021")
import auction_dispersion_round7 as M  # noqa: E402
import har_x_round9 as H  # noqa: E402
import qhar_round9 as Q  # noqa: E402

torch.set_num_threads(1)
TAUS = M.TAUS
REFIT = 66
LEV = ["neg1", "neg5", "semivar5"]


def walk_loc(d, cols, target, start):
    """OLS conditional mean + training-residual quantiles (unconditional shape)."""
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    Qm = np.full((len(d), len(TAUS)), np.nan)
    i = int((d["date"] < start).sum())
    while i < len(d):
        m = ok.copy()
        m[i:] = False
        if m.sum() < 150:
            break
        b = H.ols(X[m], y[m])
        res = y[m] - np.column_stack([np.ones(m.sum()), X[m]]) @ b
        qs = np.quantile(res, TAUS)
        j = min(i + REFIT, len(d))
        mu = np.column_stack([np.ones(j - i), X[i:j]]) @ b
        Qm[i:j] = mu.reshape(-1, 1) + qs.reshape(1, -1)
        i = j
    return Qm, ok


def score(y, Qm, mask):
    y = y[mask].reshape(-1, 1)
    q = Qm[mask]
    tau = np.array(TAUS).reshape(1, -1)
    e = y - q
    pin = np.maximum(tau * e, (tau - 1) * e)
    per = 2.0 * pin.mean(axis=1)
    lo80, hi80 = q[:, TAUS.index(0.10)], q[:, TAUS.index(0.90)]
    return {"mqs": float(per.mean()), "per": per,
            "cov80": float(np.mean((y[:, 0] >= lo80) & (y[:, 0] <= hi80))),
            "w80": float(np.mean(hi80 - lo80))}


def main():
    d = H.load()
    ARMS = {
        "L1_loc_HAR": (d, H.HAR, walk_loc),
        "L2_loc_HAR_L": (d, H.HAR + LEV, walk_loc),
        "Q1_qnt_HAR": (d, H.HAR, Q.walk_q),
        "Q2_qnt_HAR_L": (d, H.HAR + LEV, Q.walk_q),
    }
    for target in ("ly22", "ly5"):
        print(f"\n############ target = {target} ############")
        for start in ("2019-01-01", "2022-01-01"):
            truth = d[target].to_numpy(float)
            out = {}
            for name, (dd, cols, fn) in ARMS.items():
                Qm, ok = fn(dd, cols, target, start)
                out[name] = Qm
            common = np.isfinite(truth) & (d["date"] >= start).to_numpy()
            for Qm in out.values():
                common &= np.isfinite(Qm).all(1)
            print(f"  --- test from {start}   n={int(common.sum())} ---")
            if common.sum() < 50:
                print("    too few"); continue
            sc = {}
            for name in ARMS:
                s = score(truth, out[name], common)
                sc[name] = s
                print(f"    {name:<13} MQS={s['mqs']:.5f} cov80={s['cov80']:.3f} w80={s['w80']:.3f}")
            for a, b in (("Q1_qnt_HAR", "L1_loc_HAR"), ("Q2_qnt_HAR_L", "L2_loc_HAR_L"),
                         ("L2_loc_HAR_L", "L1_loc_HAR")):
                diff = sc[a]["per"] - sc[b]["per"]
                lo, hi = np.percentile(M.block_bootstrap(diff, block=10, reps=2000), [2.5, 97.5])
                gain = 100.0 * (1.0 - sc[a]["mqs"] / sc[b]["mqs"])
                star = " *" if hi < 0 else ("  (worse)" if lo > 0 else "")
                print(f"    {a} vs {b}: gain={gain:+.2f}%  CI[{lo:+.5f},{hi:+.5f}]{star}")


if __name__ == "__main__":
    main()
