"""
auction_prob_round8.py — distributional forecasting of EU ETS primary-auction
clearing prices, framed as a procurement-risk problem.

Why this and not price levels:
  Route 30 (2026-06) established the one positive result in this archive: the
  auction clearing price is predictable from two anchors (previous secondary
  close, previous auction price), beating raw persistence by +24% MAE at R^2
  0.979, robust across four splits and three auction series. What route 30 never
  delivered is a *distribution*: a compliance buyer needs an interval, not a
  point, because the obligation is quantity-and-deadline, not price.

  So the object here is P(clearing price | pre-auction information), evaluated
  with proper scoring rules and calibration, against three baselines that a
  practitioner could actually run:
    A  persistence        : clearing price = previous secondary close, quantiles
                            from the training residual distribution
    B  bridge             : linear quantile regression on the two anchors
    C  bridge + state     : + volatility, momentum, scheduled volume, calendar
    D  full               : + previous deviation (auction price vs prev close)

  Gate for a positive claim: arm D must beat arm A on MQS by >= 2.0% with a
  block-bootstrap CI excluding zero, AND keep 80% coverage within [0.75, 0.85].

Honesty note: the *level* is predictable because prices are persistent. Skill on
the level is therefore not evidence of skill on the deviation. Arms B-D are
compared against A precisely to separate those two things.
"""

import sys
import json

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\neural_model_restart_20261021")
import auction_dispersion_round7 as M  # noqa: E402

torch.set_num_threads(1)
TAUS = M.TAUS
SEEDS = [11, 22, 33]


def load():
    ROOT = r"E:\Agent员工\碳价格预测研究\独立数据归档"
    px = pd.read_csv(ROOT + r"\_legacy_202605_archive\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\01_数据\EUA_原始数据_完整版.csv",
                     usecols=["date", "eua_close_t"])
    px["date"] = pd.to_datetime(px["date"])
    px = px.dropna().sort_values("date").reset_index(drop=True)
    px["logp"] = np.log(px["eua_close_t"].astype(float))
    px["r"] = px["logp"].diff()
    px["vol22"] = px["r"].rolling(22).std()
    px["vol63"] = px["r"].rolling(63).std()
    px["ret5"] = px["logp"].diff(5)
    px["ret22"] = px["logp"].diff(22)

    a = pd.read_csv(ROOT + r"\EUA\pro_response_bridge_20261011\private_auction_base_reviewed\auction_prices_local.csv")
    a["date"] = pd.to_datetime(a["auction_date"])
    a = a.sort_values("date").reset_index(drop=True)
    a["logP"] = np.log(a["auction_price_eur_per_tco2"].astype(float))

    cur = pd.merge_asof(a[["date"]], px[["date", "logp", "vol22", "vol63", "ret5", "ret22"]],
                        on="date", direction="backward")
    # STRICT PRE-AUCTION STATE. The auction closes around 11:00, so same-day close
    # based vol/ret would be future information. Lag them one trading day.
    for c in ("vol22", "vol63", "ret5", "ret22"):
        px[c] = px[c].shift(1)
    state = pd.merge_asof(a[["date"]], px[["date", "vol22", "vol63", "ret5", "ret22"]],
                          on="date", direction="backward")
    prev = pd.merge_asof(a[["date"]], px[["date", "logp"]], on="date",
                         direction="backward", allow_exact_matches=False)
    a["close_log"] = cur["logp"].values
    a["close_prev_log"] = prev["logp"].values
    for c in ("vol22", "vol63", "ret5", "ret22"):
        a[c] = state[c].values

    a["logV"] = np.log(a["V"].astype(float))
    a["days_since"] = a["date"].diff().dt.days
    a["dow"] = a["date"].dt.dayofweek
    a["m_sin"] = np.sin(2 * np.pi * a["date"].dt.month / 12)
    a["m_cos"] = np.cos(2 * np.pi * a["date"].dt.month / 12)
    a["ser_DE"] = (a["auction_series"] == "DE").astype(float)
    a["ser_PL"] = (a["auction_series"] == "PL").astype(float)

    # lagged auction-side information (strictly previous auction)
    a["logP_prev"] = a["logP"].shift(1)
    a["dev_prev"] = (a["logP"] - a["close_prev_log"]).shift(1)
    a["dev"] = a["logP"] - a["close_prev_log"]          # target (deviation)
    a = a.dropna(subset=["close_log", "close_prev_log", "vol22", "logP_prev", "dev_prev"]).reset_index(drop=True)
    a["res_persist"] = a["logP"] - a["close_prev_log"]  # persistence residual
    return a


BASE_A = ["close_prev_log"]
BASE_B = ["close_prev_log", "logP_prev"]
STATE = ["vol22", "vol63", "ret5", "ret22", "logV", "days_since", "m_sin", "m_cos", "ser_DE", "ser_PL"]


def qr_fit_predict(Xtr, ytr, Xte, seed, linear=True, nh=32, epochs=200):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = M.QRNet(Xtr.shape[1], nq=len(TAUS), nh=nh, linear=linear)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            q = net(Xt[idx])
            e = yt[idx] - q
            loss = torch.maximum(torch.tensor(TAUS) * e, (torch.tensor(TAUS) - 1) * e).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte, dtype=torch.float32)).numpy()


def score(y, Q):
    """Q: (n, nq) sorted quantile forecasts. Returns pinball mean and coverage."""
    y = np.asarray(y).reshape(-1, 1)
    t = np.array(TAUS).reshape(1, -1)
    e = y - Q
    pin = np.maximum(t * e, (t - 1) * e)
    mqs = float(2.0 * pin.mean())
    per = 2.0 * pin.mean(axis=1)
    lo80, hi80 = Q[:, TAUS.index(0.10)], Q[:, TAUS.index(0.90)]
    lo95, hi95 = Q[:, 0], Q[:, -1]
    return {
        "mqs": mqs, "per_day": per,
        "cov80": float(np.mean((y[:, 0] >= lo80) & (y[:, 0] <= hi80))),
        "cov95": float(np.mean((y[:, 0] >= lo95) & (y[:, 0] <= hi95))),
        "w80": float(np.mean(hi80 - lo80)),
    }


def run(a, tr_mask, te_mask, cols, arm, seeds=SEEDS):
    tr = a[tr_mask].reset_index(drop=True)
    te = a[te_mask].reset_index(drop=True)
    if arm == "A":
        # empirical persistence: quantiles of training residual
        res = tr["res_persist"].values
        qs = np.quantile(res, TAUS)
        Q = np.tile(te["close_prev_log"].values.reshape(-1, 1), (1, len(TAUS))) + qs.reshape(1, -1)
        Q = np.sort(Q, axis=1)
        return [score(te["logP"].values, Q)]
    Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    # model the DEVIATION from the tradable anchor, then add the anchor back.
    # Same target as arm A, so the comparison isolates the features, not the level.
    outs = []
    for s in seeds:
        Qd = qr_fit_predict(Xtr, tr["dev"].to_numpy(float), Xte, s)
        Q = np.sort(te["close_prev_log"].to_numpy(float).reshape(-1, 1) + Qd, axis=1)
        outs.append(score(te["logP"].to_numpy(float), Q))
    return outs


def agg(outs):
    return {k: float(np.mean([o[k] for o in outs])) for k in ("mqs", "cov80", "cov95", "w80")}


def main():
    a = load()
    print(f"auctions={len(a)}  {a['date'].min().date()} -> {a['date'].max().date()}")
    report = {}
    for tag, cut in (("2023", "2023-01-01"), ("2024", "2024-01-01")):
        cut = pd.Timestamp(cut)
        trm, tem = a["date"] < cut, a["date"] >= cut
        print(f"\n========== split: train<{cut.date()}  n_tr={trm.sum()}  n_te={tem.sum()} ==========")
        res = {}
        for name, cols, arm in (
            ("A_persist", None, "A"),
            ("B_bridge", BASE_B, "B"),
            ("C_state", BASE_B + STATE, "C"),
            ("D_full", BASE_B + STATE + ["dev_prev"], "D"),
        ):
            outs = run(a, trm, tem, cols, arm)
            ag = agg(outs)
            res[name] = ag
            gain = 100.0 * (1.0 - ag["mqs"] / res["A_persist"]["mqs"])
            print(f"  {name:<10} MQS={ag['mqs']:.6f}  cov80={ag['cov80']:.3f}  cov95={ag['cov95']:.3f}"
                  f"  w80={ag['w80']:.4f}  gain_vs_A={gain:+.3f}%")
            if name != "A_persist":
                res[name]["gain"] = gain
        # bootstrap CI on the best arm vs A
        best = min((k for k in res if k != "A_persist"), key=lambda k: res[k]["mqs"])
        oa = run(a, trm, tem, None, "A")[0]
        ob = run(a, trm, tem,
                 {"B_bridge": BASE_B, "C_state": BASE_B + STATE,
                  "D_full": BASE_B + STATE + ["dev_prev"]}[best], best)
        pb = np.mean([o["per_day"] for o in ob], axis=0)
        diff = pb - oa["per_day"]
        ci = M.block_bootstrap(diff)
        lo, hi = np.percentile(ci, [2.5, 97.5]) if np.ndim(ci) else (ci[0], ci[1])
        print(f"  best arm = {best}   CI({best}-A) = [{lo:+.3e}, {hi:+.3e}]")
        report[tag] = {"arms": res, "best": best, "ci": [float(lo), float(hi)]}

    with open("auction_prob_round8.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nsaved auction_prob_round8.json")


if __name__ == "__main__":
    main()
