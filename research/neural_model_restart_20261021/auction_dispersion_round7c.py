"""
auction_dispersion_round7c.py — permutation null for the one surviving positive cell.

Observed: linear quantile model, h=5, MQS gain over BASE_LIN = +3.13% (short
baseline) / +2.15% (1,846-origin baseline), CI excludes zero, survives dropping
the 9 largest |y| days, and is NOT explained by volatility forecasting (log|y|
OOS R^2 falls when auction features are added).

A single positive cell inside a family of arms can easily be the maximum of a
noise distribution. So: build the null by destroying only the *alignment*
between auction features and the price path, keeping every other property.

Null 1 (permutation) : auction block permuted across rows. Destroys alignment
                       and the auction series' own persistence.
Null 2 (block shift) : auction block circularly shifted by a random offset of
                       >= 60 trading days. Destroys alignment but keeps the
                       auction series' autocorrelation intact — the stricter
                       null for a forecasting claim.

If the observed +3.13% sits inside either null, the cell is noise and round 7's
NEGATIVE verdict stands unqualified.
"""

import sys
import json

import numpy as np

sys.path.insert(0, r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\neural_model_restart_20261021")
import auction_dispersion_round7 as M  # noqa: E402

REPS = 25
SEED = 11
H = 5


def mqs_of(v):
    return float(2.0 * v.mean())


def gain_for(D, cols_auc, td):
    base = M.run_arm(D, M.BASE_COLS, M.CUT, td, linear=True)
    auc = M.run_arm(D, cols_auc, M.CUT, td, linear=True)
    pb = np.mean([x["per_day"] for x in base], axis=0)
    pa = np.mean([x["per_day"] for x in auc], axis=0)
    return 100.0 * (1.0 - mqs_of(pa) / mqs_of(pb)), pb, pa


def main():
    global SEED
    p = M.load_price()
    a = M.load_auction(p)
    f = M.auction_features(p, a, same_day=False)
    D = M.build_frame(p, f, M.SAMPLE_START)[H]
    td = D.loc[D["__date"] >= M.CUT, "__date"].to_numpy()
    cols_auc = M.BASE_COLS + M.AUC_COLS

    obs, pb, pa = gain_for(D, cols_auc, td)
    print(f"observed h={H} linear gain = {obs:+.3f}%")

    rng = np.random.default_rng(20261021)
    auc_block = D[M.AUC_COLS].to_numpy(float)
    n = len(D)

    perm_gains, shift_gains = [], []
    for r in range(REPS):
        Dp = D.copy()
        Dp[M.AUC_COLS] = auc_block[rng.permutation(n)]
        g, _, _ = gain_for(Dp, cols_auc, td)
        perm_gains.append(g)

        Ds = D.copy()
        off = int(rng.integers(60, n - 60))
        Ds[M.AUC_COLS] = np.roll(auc_block, off, axis=0)
        g2, _, _ = gain_for(Ds, cols_auc, td)
        shift_gains.append(g2)
        print(f"  rep {r + 1:>2}/{REPS}  perm {g:+.3f}%   shift {g2:+.3f}%")

    for name, arr in (("perm", perm_gains), ("shift", shift_gains)):
        arr = np.array(arr)
        pval = float((np.sum(arr >= obs) + 1) / (len(arr) + 1))
        print(f"{name}: mean {arr.mean():+.3f}%  sd {arr.std():.3f}  "
              f"p95 {np.percentile(arr, 95):+.3f}%  max {arr.max():+.3f}%  perm-p {pval:.3f}")

    out = {
        "h": H, "observed_gain": obs, "reps": REPS, "seed": SEED,
        "perm": {"gains": perm_gains, "mean": float(np.mean(perm_gains)),
                 "sd": float(np.std(perm_gains)), "p95": float(np.percentile(perm_gains, 95)),
                 "p_value": float((np.sum(np.array(perm_gains) >= obs) + 1) / (REPS + 1))},
        "shift": {"gains": shift_gains, "mean": float(np.mean(shift_gains)),
                  "sd": float(np.std(shift_gains)), "p95": float(np.percentile(shift_gains, 95)),
                  "p_value": float((np.sum(np.array(shift_gains) >= obs) + 1) / (REPS + 1))},
    }
    with open("auction_dispersion_round7c.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("saved auction_dispersion_round7c.json")


if __name__ == "__main__":
    main()
