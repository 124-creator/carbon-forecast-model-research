"""
signed_round10d.py — is the signed split necessary, or does HAR just need curvature?

signed_round10c found the two signed loadings are both negative and statistically
equal (z of 1.09 and 0.30, equality not rejected). With lrv22 already in the model,
an equal-weight combination of log upside and log downside variance behaves like a
curvature term: because lrv22 is the log of the SUM of the two components, adding
their equal-weighted log average lets the fitted value respond more strongly when one
direction dominates the month.

If that is all that is happening, then the signed decomposition is not the active
ingredient, and a plain quadratic term in lrv22 should reproduce the gain. That
control decides what the paper can claim.

  M0  HAR                        lrv1 lrv5 lrv22
  M1  S-HAR                      lrv1 lrv5 dn22 up22      (replace the log-sum)
  M2  HAR + curvature            lrv1 lrv5 lrv22 lrv22^2
  M3  HAR + signed pair          lrv1 lrv5 lrv22 dn22 up22
  M4  HAR + signed average       lrv1 lrv5 lrv22 (dn22+up22)

M4 is the one-parameter version implied by the equality test, so it is the minimal
model if the mechanism is one-sidedness rather than asymmetry.
"""

import numpy as np

import har_x_round9 as H

REFIT = 22


def add_signed(d):
    d = d.copy()
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    d["dn22"] = np.log(neg.rolling(22).mean().clip(lower=1e-14))
    d["up22"] = np.log(pos.rolling(22).mean().clip(lower=1e-14))
    d["gm22"] = d["dn22"] + d["up22"]
    d["lrv22_sq"] = d["lrv22"] ** 2
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
    d = add_signed(H.load())
    arms = {
        "M0_HAR": H.HAR,
        "M1_S_HAR": ["lrv1", "lrv5", "dn22", "up22"],
        "M2_curv": H.HAR + ["lrv22_sq"],
        "M3_pair": H.HAR + ["dn22", "up22"],
        "M4_avg": H.HAR + ["gm22"],
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
            base = per["M0_HAR"].mean()
            print(f"  --- test from {start}   n={int(m.sum())} ---")
            for n in arms:
                q = per[n].mean()
                lo, hi = np.percentile(block_ci(per[n] - per["M0_HAR"]), [2.5, 97.5])
                star = " *" if hi < 0 else ("  (worse)" if lo > 0 else "")
                print(f"    {n:<10} QLIKE={q:.5f}  gain={100*(1-q/base):+.2f}%  "
                      f"CI[{lo:+.6f},{hi:+.6f}]{star}")
            for a, b in (("M1_S_HAR", "M2_curv"), ("M3_pair", "M2_curv"),
                         ("M4_avg", "M2_curv")):
                diff = per[a] - per[b]
                lo, hi = np.percentile(block_ci(diff), [2.5, 97.5])
                print(f"    {a} vs {b}: {100*(1-per[a].mean()/per[b].mean()):+.2f}%  "
                      f"CI[{lo:+.6f},{hi:+.6f}]")


if __name__ == "__main__":
    main()
