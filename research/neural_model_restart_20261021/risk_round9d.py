"""
risk_round9d.py — the clean asymmetry specification.

risk_round9c found that the semivar5 arm beats HAR under QLIKE in three of four
cells, but its "HAR_DU" arm lost. That loss was a specification error, not evidence
against asymmetry: replacing lrv5 with two free components leaves both collinear
with lrv22 and both coefficients turned negative.

The interpretable way to carry asymmetry is to keep the HAR levels and add the
log ratio of downside to upside variance,

    asym = log(down5) - log(up5)

which is scale-free: it says how much of the recent variation came from falls
rather than rises, holding the overall level fixed. A positive coefficient means
downside variation forecasts more future variance than upside variation of the
same size, which is the leverage effect.

Arms:
  HAR     lrv1 lrv5 lrv22
  HAR_L   + semivar5 (the arm that worked)
  HAR_A   + asym (5-day log ratio)
  HAR_A22 + asym22 (22-day log ratio)
  HAR_AA  + both
Scored with QLIKE against the realised variance each target forecasts, block
bootstrap on the per-observation loss difference.
"""

import numpy as np

import har_x_round9 as H

LEV = ["neg1", "neg5", "semivar5"]
REFIT = 22


def add_asym(d):
    d = d.copy()
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    for w in (5, 22):
        up = pos.rolling(w).mean().clip(lower=1e-14)
        dn = neg.rolling(w).mean().clip(lower=1e-14)
        d[f"asym{w}"] = np.log(dn) - np.log(up)
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


def ql(v, y):
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
    d = add_asym(H.load())
    arms = {
        "HAR": H.HAR,
        "HAR_L": H.HAR + LEV,
        "HAR_A": H.HAR + ["asym5"],
        "HAR_A22": H.HAR + ["asym22"],
        "HAR_AA": H.HAR + ["asym5", "asym22"],
    }
    for target, proxy in (("ly5", "y5"), ("ly22", "y22")):
        print(f"\n############ target = {target}  scored against {proxy} ############")
        for start in ("2019-01-01", "2022-01-01"):
            fc = {n: walk(d, c, target, start) for n, c in arms.items()}
            m = np.isfinite(d[proxy]).to_numpy() & (d["date"] >= start).to_numpy()
            for v in fc.values():
                m &= np.isfinite(v)
            y = d[proxy].to_numpy()[m]
            per = {n: ql(fc[n][m], y) for n in arms}
            print(f"  --- test from {start}   n={int(m.sum())} ---")
            for n in arms:
                print(f"    {n:<8} QLIKE={per[n].mean():.5f}")
            for a in ("HAR_L", "HAR_A", "HAR_A22", "HAR_AA"):
                diff = per[a] - per["HAR"]
                lo, hi = np.percentile(block_ci(diff), [2.5, 97.5])
                gain = 100.0 * (1.0 - per[a].mean() / per["HAR"].mean())
                star = " *" if hi < 0 else ("  (worse)" if lo > 0 else "")
                print(f"    {a} vs HAR  gain={gain:+.2f}%  CI[{lo:+.6f},{hi:+.6f}]{star}")
    # full-sample coefficient on asym, with a Newey-West style t on the last refit
    print("\nfull-sample coefficient on asymmetry (all rows, levels not refit):")
    for target in ("ly5", "ly22"):
        for nm, cols in (("asym5", H.HAR + ["asym5"]),
                         ("asym22", H.HAR + ["asym22"])):
            X = d[cols].to_numpy(float)
            ok = np.isfinite(d[target]).to_numpy() & np.isfinite(X).all(1)
            D = np.column_stack([np.ones(ok.sum()), X[ok]])
            b, *_ = np.linalg.lstsq(D, d[target].to_numpy(float)[ok], rcond=None)
            print(f"    {target:<5} {nm:<6} beta={b[-1]:+.5f}")


if __name__ == "__main__":
    main()
