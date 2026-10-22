"""
risk_round9e.py — parameter-count placebo. Is the HAR_L gain information, or just
six extra degrees of freedom?

risk_round9/9c found that adding the six-column leverage block (neg1, neg5,
semivar5) improves QLIKE in three of four cells with bootstrap CIs excluding zero.
But risk_round9d showed that the interpretable asymmetry measure, the log ratio of
downside to upside variance, has a coefficient of roughly zero and adds nothing.
That is a warning: the gain may come from six extra parameters adapting to the
recent level, not from any asymmetry.

The test is a placebo with the same parameter count and the same marginal
distributions. The six leverage columns are permuted across rows as a single block,
which preserves each column's distribution and the correlation structure among
them, and destroys only their alignment with the target. Twenty independent
permutations give a null distribution for the QLIKE gain of six placebo columns.

If the real gain sits inside that null distribution, the effect is flexibility.
"""

import numpy as np

import har_x_round9 as H

LEV = ["neg1", "neg5", "semivar5"]
REFIT = 22
REPS = 20


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


def qlike(v, y):
    r = np.clip(v, 1e-14, None) / np.clip(y, 1e-14, None)
    return float(np.mean(r - np.log(r) - 1.0))


def main():
    d = H.load()
    rng = np.random.default_rng(20261021)
    for target, proxy in (("ly5", "y5"), ("ly22", "y22")):
        print(f"\n############ target = {target}  scored against {proxy} ############")
        for start in ("2019-01-01", "2022-01-01"):
            m0 = np.isfinite(d[proxy]).to_numpy() & (d["date"] >= start).to_numpy()
            v_har = walk(d, H.HAR, target, start)
            v_lev = walk(d, H.HAR + LEV, target, start)
            m = m0 & np.isfinite(v_har) & np.isfinite(v_lev)
            y = d[proxy].to_numpy()[m]
            q_har = qlike(v_har[m], y)
            q_lev = qlike(v_lev[m], y)
            real = 100.0 * (1.0 - q_lev / q_har)
            print(f"  --- test from {start}   n={int(m.sum())} ---")
            print(f"    HAR={q_har:.5f}  HAR_L={q_lev:.5f}  observed gain={real:+.2f}%")
            block = d[LEV].to_numpy(float)
            gains = []
            for rep in range(REPS):
                perm = rng.permutation(len(d))
                dd = d.copy()
                for k, c in enumerate(LEV):
                    dd[c + "_pl"] = block[perm, k]
                v_pl = walk(dd, H.HAR + [c + "_pl" for c in LEV], target, start)
                mm = m & np.isfinite(v_pl)
                g = 100.0 * (1.0 - qlike(v_pl[mm], d[proxy].to_numpy()[mm])
                             / qlike(v_har[mm], d[proxy].to_numpy()[mm]))
                gains.append(g)
            gains = np.array(gains)
            pct = 100.0 * float(np.mean(gains >= real))
            print(f"    placebo (same 6 columns, shuffled): mean={gains.mean():+.2f}% "
                  f"sd={gains.std():.2f} min={gains.min():+.2f} max={gains.max():+.2f}")
            print(f"    real gain beats {100-pct:.0f}% of placebos  -> p={pct/100:.2f}")


if __name__ == "__main__":
    main()
