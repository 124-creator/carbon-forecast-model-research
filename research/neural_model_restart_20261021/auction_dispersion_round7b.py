"""
auction_dispersion_round7b.py — adjudicate the one contradictory cell left by round 7.

Round 7 (MLP primary) said NEGATIVE. But the corrected linear control produced
h=5 MQS gain +3.13% with a block-bootstrap CI that excludes zero, and coverage
moved the *right* way (0.617 -> 0.686), opposite to the MLP arm. A negative
verdict is only honest if that cell is resolved, not waved away.

Three checks:
  A. Small-sample?  Compare AUC_LIN against BASE_LIN trained on 1,846 origins
     (2013+) instead of 562. If the gain survives, it is not training size.
  B. Tail-driven?   Recompute MQS after dropping test days whose |y| sits in the
     top 1%. A few crisis days must not carry the result.
  C. Channel?       Fit OLS of log|y| (and sigma^2 for QLIKE) on BASE vs
     BASE+AUC. If auction dispersion only improves the *width*, the effect is a
     volatility channel, not a return-distribution channel.
"""

import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\neural_model_restart_20261021")
import auction_dispersion_round7 as M  # noqa: E402


def trimmed_mqs(per_day, y, keep):
    return float(2.0 * per_day[keep].mean())


def vol_channel(D, cols, cut):
    """OLS of log|y| ; report out-of-sample R^2 and QLIKE for sigma^2."""
    tr, va, te = M.split_idx(D, cut)
    fit_idx = np.concatenate([tr, va])
    X = D[cols].to_numpy(float)
    mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0) + 1e-9
    X = (X - mu) / sd
    y = D["__y"].to_numpy(float)
    ly = np.log(np.abs(y) + 1e-8)
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A[fit_idx], ly[fit_idx], rcond=None)
    pred = A[te] @ beta
    sse = np.sum((ly[te] - pred) ** 2)
    sst = np.sum((ly[te] - ly[te].mean()) ** 2)
    oos_r2 = float(1.0 - sse / sst)
    sig2 = np.exp(2.0 * pred)
    y2 = y[te] ** 2
    qlike = float(np.mean(y2 / sig2 - np.log(y2 / sig2) - 1.0))
    # rank correlation between the *fitted* conditional scale and realized |y|
    fitted_scale = np.exp(pred)
    rho = float(np.corrcoef(pd.Series(fitted_scale).rank(), pd.Series(np.abs(y[te])).rank())[0, 1])
    return {"oos_r2_logabsy": oos_r2, "qlike": qlike, "spearman_fit_vs_absy": rho}


def main():
    p = M.load_price()
    a = M.load_auction(p)
    f = M.auction_features(p, a, same_day=False)
    F = M.build_frame(p, f, M.SAMPLE_START)
    F_long = M.build_frame(p, pd.DataFrame(index=p.index), M.LONG_START)
    cols_auc = M.BASE_COLS + M.AUC_COLS

    report = {}
    for h in M.HORIZONS:
        print(f"\n=== h={h} ===")
        D = F[h]
        td = D.loc[D["__date"] >= M.CUT, "__date"].to_numpy()
        te = D.index[D["__date"].isin(td)].to_numpy()
        y = D["__y"].to_numpy()[te]

        rb = M.run_arm(D, M.BASE_COLS, M.CUT, td)
        ra = M.run_arm(D, cols_auc, M.CUT, td)
        rbl = M.run_arm(D, M.BASE_COLS, M.CUT, td, linear=True)
        ral = M.run_arm(D, cols_auc, M.CUT, td, linear=True)
        rbll = M.run_arm(F_long[h], M.BASE_COLS, M.CUT, td, linear=True)

        pb = np.mean([x["per_day"] for x in rb], axis=0)
        pa = np.mean([x["per_day"] for x in ra], axis=0)
        pbl = np.mean([x["per_day"] for x in rbl], axis=0)
        pal = np.mean([x["per_day"] for x in ral], axis=0)
        pbll = np.mean([x["per_day"] for x in rbll], axis=0)

        # ---- A. small sample
        mqs = lambda v: float(2.0 * v.mean())
        gain_lin = 100.0 * (1.0 - mqs(pal) / mqs(pbl))
        gain_lin_long = 100.0 * (1.0 - mqs(pal) / mqs(pbll))
        print(f"A. linear  BASE={mqs(pbl):.6f}  BASE_LONG(n_tr={rbll[0]['n_tr']})={mqs(pbll):.6f}  AUC={mqs(pal):.6f}")
        print(f"   gain vs short BASE = {gain_lin:+.3f}%   gain vs long BASE = {gain_lin_long:+.3f}%")

        # ---- B. tails
        cut_abs = np.quantile(np.abs(y), 0.99)
        keep = np.abs(y) <= cut_abs
        g_mlp_t = 100.0 * (1.0 - mqs(pa[keep]) / mqs(pb[keep]))
        g_lin_t = 100.0 * (1.0 - mqs(pal[keep]) / mqs(pbl[keep]))
        g_lin_t_long = 100.0 * (1.0 - mqs(pal[keep]) / mqs(pbll[keep]))
        n_drop = int((~keep).sum())
        print(f"B. drop {n_drop} tail days (|y|>{cut_abs:.4f}): MLP {g_mlp_t:+.3f}%  LIN {g_lin_t:+.3f}%  LIN_vs_LONG {g_lin_t_long:+.3f}%")

        # ---- C. volatility channel
        vb = vol_channel(D, M.BASE_COLS, M.CUT)
        va = vol_channel(D, cols_auc, M.CUT)
        print(f"C. log|y| OOS R2: BASE {vb['oos_r2_logabsy']:+.4f} -> AUC {va['oos_r2_logabsy']:+.4f}"
              f"   QLIKE: {vb['qlike']:.4f} -> {va['qlike']:.4f}")
        print(f"   rank corr(fitted scale, |y|): BASE {vb['spearman_fit_vs_absy']:+.3f} -> AUC {va['spearman_fit_vs_absy']:+.3f}")

        report[h] = {
            "mqs_linear_base": mqs(pbl), "mqs_linear_base_long": mqs(pbll), "mqs_linear_auc": mqs(pal),
            "n_tr_long": rbll[0]["n_tr"],
            "gain_linear_vs_short_base": gain_lin,
            "gain_linear_vs_long_base": gain_lin_long,
            "n_tail_days_dropped": n_drop, "tail_threshold": float(cut_abs),
            "gain_mlp_trimmed": g_mlp_t, "gain_linear_trimmed": g_lin_t,
            "gain_linear_trimmed_vs_long": g_lin_t_long,
            "vol_base": vb, "vol_auc": va,
        }

    with open("auction_dispersion_round7b.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\nsaved auction_dispersion_round7b.json")


if __name__ == "__main__":
    main()
