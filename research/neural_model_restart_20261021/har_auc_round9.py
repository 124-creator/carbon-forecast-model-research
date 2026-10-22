"""
har_auc_round9.py — does primary-auction pressure carry ex-ante information about
secondary-market volatility beyond HAR and its leverage extension?

Round 9 found two things:
  1. energy features significantly HURT (overfit), so no spillover story;
  2. downside asymmetry significantly HELPS 22-day volatility forecasts in both
     test windows, which is the first consistent improvement over a real
     benchmark in this archive.

A HAR+leverage paper is too thin to be a contribution, so this tests the one
channel that is both carbon-specific and unavailable to anyone without the
auction archive: scheduled primary supply and its absorption.

Crucially every auction feature is EX-ANTE at t: only auctions strictly before t
contribute. The cover ratio of an auction held today is not known when forecasting
volatility for the following days, so it enters only after that auction date.

Arms (lean by design — the energy arm died of parameter count):
  A  HAR
  B  HAR + LEV
  C  HAR + LEV + AUC
  D  HAR + AUC
AUC = cover ratio of the last completed auction, its log volume, days since it,
and the number of auctions in the past 30 days.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\neural_model_restart_20261021")
import har_x_round9 as H  # noqa: E402

AUC = (r"E:\Agent员工\碳价格预测研究\独立数据归档\EUA\pro_response_bridge_20261011"
       r"\private_auction_base_reviewed\auction_prices_local.csv")


def add_auction(d):
    a = pd.read_csv(AUC)
    a["adate"] = pd.to_datetime(a["auction_date"])
    a = a.sort_values("adate").reset_index(drop=True)
    a["R"] = a["R"].astype(float)
    a["logV"] = np.log(a["V"].astype(float))
    a = a[["adate", "R", "logV"]].rename(columns={"R": "auc_R", "logV": "auc_logV"})
    # strictly previous auction: allow_exact_matches=False blocks same-day auctions
    d = pd.merge_asof(d, a, left_on="date", right_on="adate",
                      direction="backward", allow_exact_matches=False)
    d["auc_days"] = (d["date"] - d["adate"]).dt.days
    # auctions held in the last 30 calendar days (completed only)
    cnt = []
    for t in d["date"].to_numpy():
        cnt.append(int(((a["adate"].to_numpy() <= t) &
                        (a["adate"].to_numpy() > t - np.timedelta64(30, "D"))).sum()))
    d["auc_n30"] = cnt
    d = d.drop(columns=["adate"])
    return d


LEV = ["neg1", "neg5", "semivar5"]
AUCX = ["auc_R", "auc_logV", "auc_days", "auc_n30"]
ARMS = {
    "A_HAR": H.HAR,
    "B_HAR_LEV": H.HAR + LEV,
    "C_HAR_LEV_AUC": H.HAR + LEV + AUCX,
    "D_HAR_AUC": H.HAR + AUCX,
}


def main():
    d = add_auction(H.load())
    n_auc = d["auc_R"].notna().sum()
    print(f"auction features available on {n_auc} of {len(d)} days")
    print(d[["auc_R", "auc_days", "auc_n30"]].describe().round(3).to_string())

    for target in ("ly22", "ly5"):
        print(f"\n############ target = {target} ############")
        for start in ("2019-01-01", "2022-01-01"):
            truth = d[target].to_numpy(float)
            preds = {n: H.walk_forward(d, c, target, start) for n, c in ARMS.items()}
            common = np.isfinite(truth) & (d["date"] >= start).to_numpy()
            for p in preds.values():
                common &= np.isfinite(p)
            print(f"  --- test from {start}   n={int(common.sum())} ---")
            if common.sum() < 50:
                print("    too few observations")
                continue
            base = None
            for name in ARMS:
                e2 = (truth[common] - preds[name][common]) ** 2
                r2 = 1.0 - e2.mean() / np.var(truth[common])
                if name == "A_HAR":
                    base = e2
                    print(f"    {name:<15} OOS_R2={r2:+.4f} MSE={e2.mean():.5f}")
                    continue
                diff = e2 - base
                lo, hi = np.percentile(H.block_ci(diff), [2.5, 97.5])
                star = " *" if hi < 0 else ("  (worse)" if lo > 0 else "")
                print(f"    {name:<15} OOS_R2={r2:+.4f} MSE={e2.mean():.5f}"
                      f"  dMSE={diff.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}]{star}")


if __name__ == "__main__":
    main()
