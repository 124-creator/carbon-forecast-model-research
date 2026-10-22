"""
auction_prob_round8b.py — expanding-window walk-forward for the auction
distributional forecast, because two fixed splits disagree in sign.

Round 8 gave +10.63% MQS (CI excluding zero) on the train<2024 split and -4.86%
on the train<2023 split. Two splits cannot distinguish "real skill that grows
with training sample" from "one lucky window". This script refits at every fold
and pools the per-auction losses, so the verdict rests on the pooled
distribution rather than on a chosen cut date.

Design:
  - auctions ordered by date; fold k trains on everything before the fold start
    and tests on the next 120 auctions; walk forward to the end.
  - arms: A persistence (training-residual quantiles), B bridge, C bridge+state.
  - pooled MQS gain and block bootstrap over the pooled per-auction differences.
  - per-fold gains reported so a single fold cannot carry the result.
"""

import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\neural_model_restart_20261021")
import auction_prob_round8 as R8  # noqa: E402

MIN_TRAIN = 500
TEST_BLOCK = 120
SEEDS = [11, 22]


def main():
    a = R8.load()
    n = len(a)
    starts = list(range(MIN_TRAIN, n - 40, TEST_BLOCK))
    print(f"auctions={n}  folds={len(starts)}  min_train={MIN_TRAIN}  block={TEST_BLOCK}")

    rows = []
    for k, s in enumerate(starts):
        e = min(s + TEST_BLOCK, n)
        trm = np.zeros(n, bool); trm[:s] = True
        tem = np.zeros(n, bool); tem[s:e] = True
        fold = {"fold": k, "start": str(a["date"].iloc[s].date()), "end": str(a["date"].iloc[e - 1].date()),
                "n_tr": int(trm.sum()), "n_te": int(tem.sum())}
        base = R8.run(a, trm, tem, None, "A")
        pa = base[0]
        fold["A_mqs"] = base[0]["mqs"]
        fold["A_cov80"] = base[0]["cov80"]
        for name, cols in (("B", R8.BASE_B), ("C", R8.BASE_B + R8.STATE)):
            outs = R8.run(a, trm, tem, cols, name, seeds=SEEDS)
            pm = np.mean([o["per_day"] for o in outs], axis=0)
            fold[f"{name}_mqs"] = float(pm.mean())
            fold[f"{name}_cov80"] = float(np.mean([o["cov80"] for o in outs]))
            fold[f"{name}_gain"] = 100.0 * (1.0 - fold[f"{name}_mqs"] / fold["A_mqs"])
            fold[f"{name}_per"] = pm
        fold["A_per"] = pa["per_day"]
        rows.append(fold)
        print(f"  fold {k}  {fold['start']}..{fold['end']}  n_tr={fold['n_tr']:>4}  "
              f"A={fold['A_mqs']:.5f}(cov {fold['A_cov80']:.2f})  "
              f"B {fold['B_gain']:+7.2f}%  C {fold['C_gain']:+7.2f}%  (C cov {fold['C_cov80']:.2f})")

    print("\n" + "=" * 76)
    for name in ("B", "C"):
        pa = np.concatenate([r["A_per"] for r in rows])
        pc = np.concatenate([r[f"{name}_per"] for r in rows])
        mqs_a, mqs_c = pa.mean(), pc.mean()
        gain = 100.0 * (1.0 - mqs_c / mqs_a)
        diff = pc - pa
        ci = R8.M.block_bootstrap(diff, block=5, reps=2000)
        lo, hi = np.percentile(ci, [2.5, 97.5])
        wins = sum(1 for r in rows if r[f"{name}_gain"] > 0)
        print(f"arm {name}: pooled MQS A={mqs_a:.6f} {name}={mqs_c:.6f}  POOLED GAIN {gain:+.3f}%  "
              f"CI [{lo:+.3e}, {hi:+.3e}]  folds positive {wins}/{len(rows)}")
        print(f"         per-fold gains: " + "  ".join(f"{r[f'{name}_gain']:+.1f}" for r in rows))

    out = [{k: v for k, v in r.items() if not k.endswith("_per")} for r in rows]
    with open("auction_prob_round8b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nsaved auction_prob_round8b.json")


if __name__ == "__main__":
    main()
