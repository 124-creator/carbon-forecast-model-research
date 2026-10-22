"""
auction_dispersion_round7.py — does primary-auction *quantity dispersion* carry
incremental information about the EUA secondary-market return distribution?

WHY THIS TEST EXISTS
--------------------
Route `29_eex_official_auction_event_eua_route_20260609` (2026-06-09) already
tested EEX auction *level/scale* variables (count, price mean/min/max, volume,
bids, cover ratio, bidders, premium vs close) against EUA next-day interval/VaR
forecasts. Its pre-specified gate required >= 5% Winkler improvement; the best
candidate achieved +0.455% and event windows were negative. Verdict: FAIL.

That route never had within-auction *dispersion* of bid and award quantities.
The dispersion block (sd_B, sd_W, mu_B, mu_W) was only assembled in 2026-10
(`EUA/pro_response_bridge_20261011`) and is the construct behind the submitted
Empirical Economics manuscript. This script asks the narrow question the prior
route could not: does demand *heterogeneity* add distributional information
beyond price history and beyond auction levels?

PRE-REGISTERED GATE (fixed before running)
------------------------------------------
PASS  iff  MQS gain >= +2.0% at h=1 OR h=5, AND the block-bootstrap 95% CI
           upper bound of (AUC - BASE) per-day pinball is < 0 at that horizon.
Anything else is NEGATIVE and is reported as such.

DESIGN
------
- Target: y = log p[t+h] - log p[t], h in {1,5}; 15 quantiles; MQS = 2*mean pinball.
- Scale: trailing 120-day sd of daily log returns at t, times sqrt(h).
- Origins: both arms use *identical* origins, 2019-12-02 onward (auction sample).
- BASE        : 20 price-history features.
- AUC         : BASE + 13 auction-dispersion features, auctions strictly before t.
- AUC_SAMEDAY : same features but an auction on day t itself is allowed (results
                are published before the secondary close, so this is legitimate
                too); secondary reading.
- BASE_LONG   : BASE features trained from 2013 onward. Separates "less training
                data" from "auction information".
- Model: one workhorse MLP (15 quantiles, 32 hidden, tanh, dropout .05),
  AdamW lr 1e-3 wd 3e-3, clip 1, inner-val early stop, refit on train+val.
- Seeds: 11, 22, 33.
"""

import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)

ROOT = r"E:\Agent员工\碳价格预测研究\独立数据归档"
EUA_CSV = ROOT + r"\_legacy_202605_archive\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\01_数据\EUA_原始数据_完整版.csv"
AUC_CSV = ROOT + r"\EUA\pro_response_bridge_20261011\private_auction_base_reviewed\auction_prices_local.csv"

TAUS = [0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975]
HORIZONS = [1, 5]
SEEDS = [11, 22, 33]
CUT = pd.Timestamp("2023-01-02")
SAMPLE_START = pd.Timestamp("2019-12-02")
LONG_START = pd.Timestamp("2013-06-01")

BASE_COLS = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_lag10", "ret_lag20",
             "vol5", "vol10", "vol22", "vol63",
             "gap_ma5", "gap_ma22", "gap_ma63",
             "absret_ma5", "absret_ma22", "trend120",
             "dow_sin", "dow_cos", "mon_sin", "mon_cos"]

AUC_COLS = ["cvB", "cvW", "lcover", "lbidders", "lsucc", "lbid_size", "lbids_per",
            "spread_pct", "auc_premium", "days_since",
            "roll5_cvB", "roll22_cvB", "roll5_cover"]


# ---------------------------------------------------------------- data helpers
def load_price():
    df = pd.read_csv(EUA_CSV, usecols=["date", "eua_close_t"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["eua_close_t"]).sort_values("date").reset_index(drop=True)
    df["logp"] = np.log(df["eua_close_t"].astype(float))
    df["ret"] = df["logp"].diff()
    return df


def price_features(p):
    f = pd.DataFrame(index=p.index)
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret_lag{k}"] = p["ret"].shift(k)
    for w in (5, 10, 22, 63):
        f[f"vol{w}"] = p["ret"].rolling(w).std()
    for w in (5, 22, 63):
        f[f"gap_ma{w}"] = p["logp"] - p["logp"].rolling(w).mean()
    f["absret_ma5"] = p["ret"].abs().rolling(5).mean()
    f["absret_ma22"] = p["ret"].abs().rolling(22).mean()
    f["trend120"] = p["logp"] - p["logp"].shift(120)
    dow = p["date"].dt.dayofweek.astype(float)
    mon = p["date"].dt.month.astype(float)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    f["mon_sin"] = np.sin(2 * np.pi * mon / 12.0)
    f["mon_cos"] = np.cos(2 * np.pi * mon / 12.0)
    return f[BASE_COLS].astype(float)


def load_auction(p):
    a = pd.read_csv(AUC_CSV)
    a["auc_date"] = pd.to_datetime(a["auction_date"])
    a = a.sort_values("auc_date").reset_index(drop=True)
    out = pd.DataFrame({"auc_date": a["auc_date"]})
    out["cvB"] = np.log(a["sd_B"] / a["mu_B"])
    out["cvW"] = np.log(a["sd_W"] / a["mu_W"])
    out["lcover"] = np.log(a["R"])
    out["lbidders"] = np.log(a["N"])
    out["lsucc"] = np.log(a["S"])
    out["lbid_size"] = np.log(a["mean_bid_size"])
    out["lbids_per"] = np.log(a["mean_bids_per_bidder"])
    out["spread_pct"] = np.log1p(a["bid_price_spread_eur_per_tco2"] / a["auction_price_eur_per_tco2"])
    prev = pd.merge_asof(out[["auc_date"]], p[["date", "logp"]].rename(columns={"date": "auc_date"}).sort_values("auc_date"),
                         on="auc_date", direction="backward", allow_exact_matches=False)
    out["auc_premium"] = np.log(a["auction_price_eur_per_tco2"].astype(float).values) - prev["logp"].values
    return out


def auction_features(p, a, same_day):
    """Most recent auction block attached to every trading day."""
    left = p[["date"]].rename(columns={"date": "auc_date"}).copy()
    left["tpos"] = np.arange(len(left))
    right = a.assign(auction_match=a["auc_date"]).sort_values("auc_date")
    m = pd.merge_asof(left.sort_values("auc_date"), right, on="auc_date",
                      direction="backward", allow_exact_matches=same_day)
    m = m.sort_values("tpos").reset_index(drop=True)
    matched = pd.to_datetime(m.pop("auction_match"))
    m["days_since"] = np.log1p((p["date"] - matched).dt.days.clip(lower=0).astype(float))
    f = pd.DataFrame(index=p.index)
    for c in ["cvB", "cvW", "lcover", "lbidders", "lsucc", "lbid_size", "lbids_per",
              "spread_pct", "auc_premium", "days_since"]:
        f[c] = m[c].astype(float).values
    f["roll5_cvB"] = f["cvB"].rolling(5).mean()
    f["roll22_cvB"] = f["cvB"].rolling(22).mean()
    f["roll5_cover"] = f["lcover"].rolling(5).mean()
    return f[AUC_COLS].astype(float)


# ---------------------------------------------------------------- target frame
def build_frame(p, extra, start):
    base = price_features(p)
    X = pd.concat([base, extra], axis=1)
    X["__date"] = p["date"].values
    X["__logp"] = p["logp"].values
    sd = p["ret"].rolling(120).std()
    X["__sd"] = sd.values
    out = {}
    n = len(p)
    for h in HORIZONS:
        D = X.iloc[: n - h].copy()
        D["__y"] = p["logp"].shift(-h).values[: n - h] - D["__logp"].values
        D["__scale"] = D["__sd"].values * math.sqrt(h)
        D = D.replace([np.inf, -np.inf], np.nan).dropna()
        D = D[D["__date"] >= start].sort_values("__date").reset_index(drop=True)
        out[h] = D
    return out


def split_idx(D, cut):
    pre = D.index[(D["__date"] < cut).to_numpy()].to_numpy()
    n_val = max(1, int(round(0.25 * len(pre))))
    return pre[: len(pre) - n_val], pre[len(pre) - n_val:], D.index[(D["__date"] >= cut).to_numpy()].to_numpy()


# ---------------------------------------------------------------- model
class QRNet(nn.Module):
    def __init__(self, nin, nq=len(TAUS), nh=32, linear=False):
        super().__init__()
        if linear:
            self.body = nn.Linear(nin, nq)
        else:
            self.body = nn.Sequential(nn.Linear(nin, nh), nn.Tanh(), nn.Dropout(0.05), nn.Linear(nh, nq))

    def forward(self, x):
        return torch.sort(self.body(x), dim=1).values


def pinball(y, q, taus):
    d = y[:, None] - q
    return np.maximum(taus[None, :] * d, (taus[None, :] - 1.0) * d)


def fit_predict(Xtr, ytr, Xva, yva, Xte, seed, epochs_max=120, patience=20, min_epochs=30,
                linear=False):
    taus_t = torch.tensor(TAUS, dtype=torch.float32)

    def loss_of(model, x, y):
        d = y[:, None] - model(x)
        return torch.maximum(taus_t[None, :] * d, (taus_t[None, :] - 1.0) * d).mean()

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = QRNet(Xtr.shape[1], linear=linear)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-3)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)
    best, best_ep, wait = np.inf, 0, 0
    for ep in range(epochs_max):
        model.train()
        opt.zero_grad()
        loss_of(model, xt, yt).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_of(model, xv, yv).item()
        if vl < best - 1e-7:
            best, best_ep, wait = vl, ep + 1, 0
        else:
            wait += 1
            if wait >= patience and ep + 1 >= min_epochs:
                break
    n_refit = max(best_ep, 1)

    Xa = np.vstack([Xtr, Xva])
    ya = np.concatenate([ytr, yva])
    torch.manual_seed(seed)
    model = QRNet(Xa.shape[1], linear=linear)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-3)
    xa = torch.tensor(Xa, dtype=torch.float32)
    ya_t = torch.tensor(ya, dtype=torch.float32)
    for _ in range(n_refit):
        model.train()
        opt.zero_grad()
        loss_of(model, xa, ya_t).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, dtype=torch.float32)).numpy()
    return out, best_ep, n_refit


def evaluate(y, q):
    ya = y[:, None]
    pv = pinball(ya, q, np.array(TAUS))
    return {
        "mqs": float(2.0 * pv.mean()),
        "cov80": float(np.mean((ya[:, 0] >= q[:, 2]) & (ya[:, 0] <= q[:, 12]))),
        "cov95": float(np.mean((ya[:, 0] >= q[:, 0]) & (ya[:, 0] <= q[:, 14]))),
        "r2": float(1.0 - np.sum((ya[:, 0] - q[:, 7]) ** 2) / np.sum((ya[:, 0] - ya.mean()) ** 2)),
        "per_day": pv.mean(axis=1),
    }


def run_arm(D, cols, cut, test_dates, train_frame=None, linear=False):
    """Train on `train_frame` (default D) but always evaluate on `test_dates` rows of D."""
    T = train_frame if train_frame is not None else D
    tr, va, _ = split_idx(T, cut)
    te = D.index[D["__date"].isin(test_dates)].to_numpy()
    assert len(te) == len(test_dates)
    Xt = T[cols].to_numpy(dtype=np.float64)
    mu, sd = Xt[tr].mean(axis=0), Xt[tr].std(axis=0) + 1e-9
    Xt = (Xt - mu) / sd
    Xd = (D[cols].to_numpy(dtype=np.float64) - mu) / sd
    yt = (T["__y"] / T["__scale"]).to_numpy(dtype=np.float64)
    res = []
    for seed in SEEDS:
        q, be, nr = fit_predict(Xt[tr], yt[tr], Xt[va], yt[va], Xd[te], seed, linear=linear)
        scale = D["__scale"].to_numpy()[te]
        m = evaluate(D["__y"].to_numpy()[te], q * scale[:, None])
        m.update(seed=seed, best_ep=be, refit_ep=nr, n_tr=len(tr), n_va=len(va), n_te=len(te))
        res.append(m)
    return res


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def marginal_screen(D, label):
    """Model-free check: does auction dispersion line up with the outcome at all?"""
    out = {}
    pre = D["__date"] < CUT
    for split, mask in (("precut", pre), ("test", ~pre)):
        sub = D[mask]
        row = {}
        for c in ["cvB", "cvW", "lcover", "lbid_size", "lbids_per", "auc_premium"]:
            row[c] = {
                "vs_abs_y": spearman(sub[c], sub["__y"].abs()),
                "vs_y": spearman(sub[c], sub["__y"]),
            }
        out[split] = row
    print(f"  [screen {label}] Spearman vs |y| (precut/test):", end=" ")
    for c in ["cvB", "cvW", "lcover"]:
        print(f"{c}={out['precut'][c]['vs_abs_y']:+.3f}/{out['test'][c]['vs_abs_y']:+.3f}", end="  ")
    print()
    return out


def block_bootstrap(diff, block=5, reps=1000, seed=20261021):
    rng = np.random.default_rng(seed)
    n = len(diff)
    out = np.empty(reps)
    for r in range(reps):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            idx.extend(range(s, min(s + block, n)))
        out[r] = diff[np.array(idx[:n])].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ("mqs", "cov80", "cov95", "r2")}


def main():
    t0 = time.time()
    p = load_price()
    a = load_auction(p)
    print(f"price rows={len(p)}  {p['date'].min().date()} -> {p['date'].max().date()}")
    print(f"auctions={len(a)}  {a['auc_date'].min().date()} -> {a['auc_date'].max().date()}")

    f_lag1 = auction_features(p, a, same_day=False)
    f_same = auction_features(p, a, same_day=True)
    F_lag1 = build_frame(p, f_lag1, SAMPLE_START)
    F_same = build_frame(p, f_same, SAMPLE_START)
    # BASE_LONG must not be polluted by auction NaNs, else dropna() silently
    # collapses it onto the 2020+ sample. Use BASE_COLS only.
    F_long = build_frame(p, pd.DataFrame(index=p.index), LONG_START)
    print(f"origin counts: short={len(F_lag1[1])} long={len(F_long[1])}")

    results = {}
    screens = {}
    for h in HORIZONS:
        print(f"\n=== horizon {h} ===")
        D = F_lag1[h]
        test_dates = D.loc[D["__date"] >= CUT, "__date"].to_numpy()
        cols_auc = BASE_COLS + AUC_COLS
        rb = run_arm(D, BASE_COLS, CUT, test_dates)
        ra = run_arm(D, cols_auc, CUT, test_dates)
        rs = run_arm(F_same[h], cols_auc, CUT, test_dates)
        rl = run_arm(F_long[h], BASE_COLS, CUT, test_dates)
        rbl = run_arm(D, BASE_COLS, CUT, test_dates, linear=True)
        ral = run_arm(D, cols_auc, CUT, test_dates, linear=True)
        screens[h] = marginal_screen(D, f"h={h}")

        ab, aa, asd, al = agg(rb), agg(ra), agg(rs), agg(rl)
        abl, aal = agg(rbl), agg(ral)
        gain = 100.0 * (1.0 - aa["mqs"] / ab["mqs"])
        gain_sd = 100.0 * (1.0 - asd["mqs"] / ab["mqs"])
        gain_long = 100.0 * (1.0 - aa["mqs"] / al["mqs"])
        gain_lin = 100.0 * (1.0 - aal["mqs"] / abl["mqs"])

        pb = np.mean([x["per_day"] for x in rb], axis=0)
        pa = np.mean([x["per_day"] for x in ra], axis=0)
        ps = np.mean([x["per_day"] for x in rs], axis=0)
        pl = np.mean([x["per_day"] for x in rl], axis=0)
        pbl = np.mean([x["per_day"] for x in rbl], axis=0)
        pal = np.mean([x["per_day"] for x in ral], axis=0)
        ci = block_bootstrap(pa - pb)
        ci_sd = block_bootstrap(ps - pb)
        ci_l = block_bootstrap(pa - pl)
        ci_lin = block_bootstrap(pal - pbl)

        results[h] = {
            "n_tr": rb[0]["n_tr"], "n_va": rb[0]["n_va"], "n_te": rb[0]["n_te"],
            "n_tr_long": rl[0]["n_tr"],
            "base": ab, "auc": aa, "auc_sameday": asd, "base_long": al,
            "base_lin": abl, "auc_lin": aal,
            "gain": gain, "gain_sameday": gain_sd, "gain_vs_base_long": gain_long,
            "gain_lin": gain_lin,
            "ci_vs_base": ci, "ci_sameday_vs_base": ci_sd, "ci_vs_base_long": ci_l,
            "ci_lin": ci_lin,
            "best_ep": [x["best_ep"] for x in ra], "refit_ep": [x["refit_ep"] for x in ra],
        }
        print(f"n_tr={rb[0]['n_tr']} n_va={rb[0]['n_va']} n_te={rb[0]['n_te']}  n_tr_long={rl[0]['n_tr']}")
        print(f"BASE        MQS={ab['mqs']:.6f} cov80={ab['cov80']:.3f} R2={ab['r2']:.4f}")
        print(f"AUC         MQS={aa['mqs']:.6f} cov80={aa['cov80']:.3f} R2={aa['r2']:.4f}  gain={gain:+.3f}%")
        print(f"AUC_SAMEDAY MQS={asd['mqs']:.6f} cov80={asd['cov80']:.3f}  gain={gain_sd:+.3f}%")
        print(f"BASE_LONG   MQS={al['mqs']:.6f} cov80={al['cov80']:.3f}  gain={gain_long:+.3f}%")
        print(f"BASE_LIN    MQS={abl['mqs']:.6f} cov80={abl['cov80']:.3f}")
        print(f"AUC_LIN     MQS={aal['mqs']:.6f} cov80={aal['cov80']:.3f}  gain={gain_lin:+.3f}%")
        print(f"CI(AUC-BASE)      = [{ci[0]:+.3e}, {ci[1]:+.3e}]")
        print(f"CI(SAMEDAY-BASE)  = [{ci_sd[0]:+.3e}, {ci_sd[1]:+.3e}]")
        print(f"CI(AUC-BASE_LONG) = [{ci_l[0]:+.3e}, {ci_l[1]:+.3e}]")
        print(f"CI(AUC_LIN-BASE_LIN) = [{ci_lin[0]:+.3e}, {ci_lin[1]:+.3e}]")

    gates = []
    for h in HORIZONS:
        r = results[h]
        ok = bool(r["gain"] >= 2.0 and r["ci_vs_base"][1] < 0.0)
        gates.append({"h": h, "gain_pct": r["gain"], "ci": r["ci_vs_base"], "pass": ok})
    verdict = "PASS" if any(g["pass"] for g in gates) else "NEGATIVE"

    out = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "gate_rule": "MQS gain >= +2.0% and block-bootstrap 95% CI upper bound of (AUC-BASE) per-day pinball < 0",
        "gates": gates,
        "results": {str(h): results[h] for h in HORIZONS},
        "marginal_screen": {str(h): screens[h] for h in HORIZONS},
        "n_auctions": int(len(a)),
        "auction_span": [str(a["auc_date"].min().date()), str(a["auc_date"].max().date())],
        "seconds": round(time.time() - t0, 1),
    }
    with open("auction_dispersion_round7.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    rows = []
    for h in HORIZONS:
        for arm in ("base", "auc", "auc_sameday", "base_long", "base_lin", "auc_lin"):
            rows.append({"h": h, "arm": arm, **(results[h][arm])})
    pd.DataFrame(rows).to_csv("auction_dispersion_round7_cells.csv", index=False)
    print(f"\nVERDICT: {verdict}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
