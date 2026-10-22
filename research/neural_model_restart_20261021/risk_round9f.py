"""
risk_round9f.py — HAR-SV: a multi-scale downside specification.

risk_round9e showed the six-column leverage block carries real information (it
beats all 20 parameter-count placebos), while risk_round9d showed the effect cannot
be reduced to a single asymmetry ratio. The block mixes a one-day term, a five-day
term and a semi-variance, so the natural interpretation is that the informative
object is the *term structure of downside variation*, mirroring the multi-scale
structure that makes HAR work in the first place.

That gives a specification rather than a feature dump:

    HAR-SV: lrv1 lrv5 lrv22  +  log downside variance at 1, 5 and 22 days

i.e. a symmetric HAR on the levels plus the same three scales computed only from
negative returns. Also tested with the upside scales added, which turns it into a
full signed decomposition and is the stricter version.

Every arm is scored with QLIKE against the realised variance it forecasts, with
block-bootstrap CIs, and the best arm is passed through the same 20-draw
parameter-count placebo as risk_round9e so the identification standard is unchanged.
"""

import numpy as np

import har_x_round9 as H

REFIT = 22
REPS = 20
DOWN = ["dn1", "dn5", "dn22"]
UP = ["up1", "up5", "up22"]


def add_scales(d):
    d = d.copy()
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    for w in (1, 5, 22):
        d[f"dn{w}"] = np.log(neg.rolling(w).mean().clip(lower=1e-14))
        d[f"up{w}"] = np.log(pos.rolling(w).mean().clip(lower=1e-14))
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
    d = add_scales(H.load())
    arms = {
        "HAR": H.HAR,
        "HAR_L": H.HAR + ["neg1", "neg5", "semivar5"],
        "HAR_SV": H.HAR + DOWN,
        "HAR_SVU": H.HAR + DOWN + UP,
    }
    rng = np.random.default_rng(20261021)
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
            for a in ("HAR_L", "HAR_SV", "HAR_SVU"):
                diff = per[a] - per["HAR"]
                lo, hi = np.percentile(block_ci(diff), [2.5, 97.5])
                gain = 100.0 * (1.0 - per[a].mean() / per["HAR"].mean())
                star = " *" if hi < 0 else ("  (worse)" if lo > 0 else "")
                print(f"    {a} vs HAR  gain={gain:+.2f}%  CI[{lo:+.6f},{hi:+.6f}]{star}")
            # placebo for whichever arm is best, to keep the identification standard
            best = min(("HAR_SV", "HAR_SVU"), key=lambda n: per[n].mean())
            extra = DOWN if best == "HAR_SV" else DOWN + UP
            block = d[extra].to_numpy(float)
            gains = []
            for rep in range(REPS):
                perm = rng.permutation(len(d))
                dd = d.copy()
                for k, c in enumerate(extra):
                    dd[c + "_pl"] = block[perm, k]
                v_pl = walk(dd, H.HAR + [c + "_pl" for c in extra], target, start)
                mm = m & np.isfinite(v_pl)
                gains.append(100.0 * (1.0 - qlike_arm(v_pl, dd, proxy, mm)
                                      / qlike_arm(fc["HAR"], d, proxy, mm)))
            gains = np.array(gains)
            real = 100.0 * (1.0 - per[best].mean() / per["HAR"].mean())
            p = 100.0 * float(np.mean(gains >= real))
            print(f"    placebo[{best}, {len(extra)} cols] mean={gains.mean():+.2f}% "
                  f"sd={gains.std():.2f} max={gains.max():+.2f}  real={real:+.2f}%  "
                  f"beats {100-p:.0f}%  p={p/100:.2f}")


def qlike_arm(v, d, proxy, m):
    return float(np.mean(ql(v[m], d[proxy].to_numpy()[m])))


if __name__ == "__main__":
    main()
