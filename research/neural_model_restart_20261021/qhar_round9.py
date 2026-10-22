"""
qhar_round9.py — distributional volatility forecasting for EU ETS.

Motivation from round 9: the only extension that survives out of sample is
downside asymmetry, and only at the 22-day horizon (dMSE negative with bootstrap
CIs excluding zero in both test windows; null at h=1 and h=5). Energy spillovers
and auction pressure are significantly harmful.

A mean improvement of ~1.2 R^2 points is too thin for a paper. What a compliance
desk or a margin model actually needs is the *distribution* of future volatility,
so this asks whether asymmetry improves the quantile forecast, not just the mean.

Arms, identical observations, 15 quantiles, walk-forward with refit every 66 days:
  Q0  climatology      training-sample quantiles of the target, no conditioning
  Q1  quantile HAR     linear quantile regression on lrv1/lrv5/lrv22
  Q2  quantile HAR-L   + downside asymmetry (neg1, neg5, semivar5)

Scored with 2 x mean pinball (MQS, equal to CRPS for quantile forecasts), 80/95
coverage, and tail pinball. A win requires Q2 to beat Q1 with a block-bootstrap
CI excluding zero; Q1 must also beat Q0 or conditioning is pointless.
"""

import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\neural_model_restart_20261021")
import auction_dispersion_round7 as M  # noqa: E402  (TAUS, QRNet, block_bootstrap)
import har_x_round9 as H  # noqa: E402

torch.set_num_threads(1)
TAUS = M.TAUS
SEEDS = [11, 22, 33]
REFIT = 66


def fit_q(Xtr, ytr, Xte, seed, epochs=400):
    torch.manual_seed(seed)
    net = M.QRNet(Xtr.shape[1], nq=len(TAUS), nh=32, linear=True)
    opt = torch.optim.AdamW(net.parameters(), lr=0.05, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    tau = torch.tensor(TAUS)
    for _ in range(epochs):
        q = net(Xt)
        e = yt - q
        loss = torch.maximum(tau * e, (tau - 1) * e).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte, dtype=torch.float32)).numpy()


def walk_q(d, cols, target, start):
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float) if cols else np.zeros((len(d), 0))
    ok = np.isfinite(y) & (np.isfinite(X).all(1) if cols else True)
    Q = np.full((len(d), len(TAUS)), np.nan)
    i = int((d["date"] < start).sum())
    while i < len(d):
        m = ok.copy()
        m[i:] = False
        if m.sum() < 150:
            break
        j = min(i + REFIT, len(d))
        if cols:
            mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
            ymu = y[m].mean()
            Xtr, Xte = (X[m] - mu) / sd, (X[i:j] - mu) / sd
            for s in SEEDS:
                Qd = fit_q(Xtr, y[m] - ymu, Xte, s)
                Q[i:j] = Q[i:j] + (ymu + np.sort(Qd, axis=1)) / len(SEEDS) if np.isfinite(Q[i:j]).any() else (ymu + np.sort(Qd, axis=1)) / len(SEEDS)
        else:
            qs = np.quantile(y[m], TAUS)
            Q[i:j] = qs.reshape(1, -1)
        i = j
    return Q, ok


def score(y, Q, mask):
    y = y[mask].reshape(-1, 1)
    q = Q[mask]
    tau = np.array(TAUS).reshape(1, -1)
    e = y - q
    pin = np.maximum(tau * e, (tau - 1) * e)
    per = 2.0 * pin.mean(axis=1)
    lo80, hi80 = q[:, TAUS.index(0.10)], q[:, TAUS.index(0.90)]
    lo95, hi95 = q[:, 0], q[:, -1]
    tail = 2.0 * pin[:, [0, 1, -2, -1]].mean(axis=1)
    return {"mqs": float(per.mean()), "per": per,
            "cov80": float(np.mean((y[:, 0] >= lo80) & (y[:, 0] <= hi80))),
            "cov95": float(np.mean((y[:, 0] >= lo95) & (y[:, 0] <= hi95))),
            "w80": float(np.mean(hi80 - lo80)), "tail": float(tail.mean())}


def main():
    d = H.load()
    LEV = ["neg1", "neg5", "semivar5"]
    ARMS = {"Q0_clim": [], "Q1_qHAR": H.HAR, "Q2_qHAR_L": H.HAR + LEV}
    for target in ("ly22", "ly5"):
        print(f"\n############ target = {target} ############")
        for start in ("2019-01-01", "2022-01-01"):
            truth = d[target].to_numpy(float)
            res = {}
            for name, cols in ARMS.items():
                Q, ok = walk_q(d, cols, target, start)
                res[name] = (Q, ok)
            common = np.isfinite(truth) & (d["date"] >= start).to_numpy()
            for Q, ok in res.values():
                common &= np.isfinite(Q).all(1)
            print(f"  --- test from {start}   n={int(common.sum())} ---")
            if common.sum() < 50:
                print("    too few"); continue
            sc = {}
            for name in ARMS:
                s = score(truth, res[name][0], common)
                sc[name] = s
                print(f"    {name:<10} MQS={s['mqs']:.5f} tail={s['tail']:.5f} "
                      f"cov80={s['cov80']:.3f} cov95={s['cov95']:.3f} w80={s['w80']:.3f}")
            for a, b in (("Q1_qHAR", "Q0_clim"), ("Q2_qHAR_L", "Q1_qHAR")):
                diff = sc[a]["per"] - sc[b]["per"]
                lo, hi = np.percentile(M.block_bootstrap(diff, block=10, reps=2000), [2.5, 97.5])
                gain = 100.0 * (1.0 - sc[a]["mqs"] / sc[b]["mqs"])
                star = " *" if hi < 0 else ""
                print(f"    {a} vs {b}: gain={gain:+.2f}%  dMQS={diff.mean():+.5f} "
                      f"CI[{lo:+.5f},{hi:+.5f}]{star}")


if __name__ == "__main__":
    main()
