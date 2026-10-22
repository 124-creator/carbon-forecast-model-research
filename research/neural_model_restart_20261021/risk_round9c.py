"""
risk_round9c.py — is the asymmetry gain statistically real, and is it asymmetry
or just the recent level?

Two things a referee will demand.

1. A confidence interval on the QLIKE difference. QLIKE is a per-observation loss,
   so the difference is bootstrapped with blocks to respect volatility clustering.

2. A specification that separates the sign of the move. semivar5 is the mean
   squared NEGATIVE return over five days, and HAR already contains lrv5, the mean
   squared return irrespective of sign. So the asymmetry arm is legitimate, but a
   cleaner test decomposes the five-day variance into a down part and an up part
   and asks whether the down part carries the information. If only the down part
   matters, that is the leverage effect. If both matter equally, the arm was just
   measuring the recent level and the interpretation fails.

Arms:
  HAR     lrv1 lrv5 lrv22
  HAR_L   + semivar5 (downside only)
  HAR_DU  + log(down variance) + log(up variance), replacing lrv5
"""

import numpy as np

import har_x_round9 as H

LEV = ["neg1", "neg5", "semivar5"]
REFIT = 22


def add_updown(d):
    d = d.copy()
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    d["up5"] = np.log(pos.rolling(5).mean().clip(lower=1e-14))
    d["down5"] = np.log(neg.rolling(5).mean().clip(lower=1e-14))
    return d


def walk(d, cols, target, start, refit=REFIT):
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float)
    v = np.full(len(d), np.nan)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
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
        v[i:j] = np.exp(np.column_stack([np.ones(j - i), X[i:j]]) @ b) * f
        i = j
    return v


def ql_per_obs(v, y):
    r = np.clip(v, 1e-14, None) / np.clip(y, 1e-14, None)
    return r - np.log(r) - 1.0


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
    d = add_updown(H.load())
    arms = {
        "HAR": H.HAR,
        "HAR_L": H.HAR + LEV,
        "HAR_DU": ["lrv1", "up5", "down5", "lrv22"],
    }
    for target, proxy in (("ly5", "y5"), ("ly22", "y22")):
        print(f"\n############ target = {target}  scored against {proxy} ############")
        for start in ("2019-01-01", "2022-01-01"):
            fc = {n: walk(d, c, target, start) for n, c in arms.items()}
            m = np.isfinite(d[proxy]).to_numpy() & (d["date"] >= start).to_numpy()
            for v in fc.values():
                m &= np.isfinite(v)
            y = d[proxy].to_numpy()[m]
            per = {n: ql_per_obs(fc[n][m], y) for n in arms}
            print(f"  --- test from {start}   n={int(m.sum())} ---")
            for n in arms:
                print(f"    {n:<8} QLIKE={per[n].mean():.5f}")
            for a, b in (("HAR_L", "HAR"), ("HAR_DU", "HAR"), ("HAR_DU", "HAR_L")):
                diff = per[a] - per[b]
                lo, hi = np.percentile(block_ci(diff), [2.5, 97.5])
                gain = 100.0 * (1.0 - per[a].mean() / per[b].mean())
                star = " *" if hi < 0 else ("  (worse)" if lo > 0 else "")
                print(f"    {a} vs {b:<6} gain={gain:+.2f}%  dQLIKE={diff.mean():+.6f} "
                      f"CI[{lo:+.6f},{hi:+.6f}]{star}")
            # does the down part dominate the up part?
            if "HAR_DU" in arms and True:
                X = d[["lrv1", "up5", "down5", "lrv22"]].to_numpy(float)
                ok = np.isfinite(d[target]).to_numpy() & np.isfinite(X).all(1)
                D = np.column_stack([np.ones(ok.sum()), X[ok]])
                b, *_ = np.linalg.lstsq(D, d[target].to_numpy(float)[ok], rcond=None)
                print(f"    full-sample coefficients  up5={b[2]:+.4f}  down5={b[3]:+.4f}"
                      f"  ratio={abs(b[3]/b[2]):.2f}x")


if __name__ == "__main__":
    main()
