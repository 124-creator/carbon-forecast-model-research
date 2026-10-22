"""
auction_gap_scan.py — feasibility scan for a POSITIVE-framed target:
the primary-secondary price gap in EU ETS auctions.

Why this object:
  - We have exhausted price-level and return-distribution forecasting. Both are
    saturated (see FINAL_REPORT_ZH.md, AUCTION_DISPERSION_ZH.md).
  - The auction gap `g = log(secondary close) - log(auction clearing price)` is a
    *spread between two markets for the same good on the same day*. Spreads can
    mean-revert even when levels are near random walks.
  - It is the quantity a compliance buyer actually trades off: bid in the primary
    auction, or buy in the secondary market. 2.6 Mt average auction size means a
    0.20 EUR/tCO2 gap is ~EUR 520k per auction, so even small predictability is
    economically material.

This script does NOT test a model. It only asks whether the object is worth a
paper: level, dispersion, persistence, seasonality, ex-ante vs ex-post drivers.
Gate for proceeding: the gap must show non-trivial persistence or ex-ante
structure, otherwise the honest answer is "no paper here either".
"""

import numpy as np
import pandas as pd

ROOT = r"E:\Agent员工\碳价格预测研究\独立数据归档"
EUA = ROOT + r"\_legacy_202605_archive\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\01_数据\EUA_原始数据_完整版.csv"
AUC = ROOT + r"\EUA\pro_response_bridge_20261011\private_auction_base_reviewed\auction_prices_local.csv"

pd.set_option("display.width", 200)


def main():
    px = pd.read_csv(EUA, usecols=["date", "eua_close_t"])
    px["date"] = pd.to_datetime(px["date"])
    px = px.dropna().sort_values("date").reset_index(drop=True)
    px["logp"] = np.log(px["eua_close_t"].astype(float))

    a = pd.read_csv(AUC)
    a["date"] = pd.to_datetime(a["auction_date"])
    a = a.sort_values("date").reset_index(drop=True)

    # secondary close on the auction day, and the previous trading day's close
    m = pd.merge_asof(a[["date"]], px[["date", "logp"]], on="date", direction="backward")
    mprev = pd.merge_asof(a[["date"]], px[["date", "logp"]], on="date",
                          direction="backward", allow_exact_matches=False)
    a["close_same"] = np.exp(m["logp"].values)
    a["close_prev"] = np.exp(mprev["logp"].values)
    a["auction_price"] = a["auction_price_eur_per_tco2"].astype(float)

    # gap definitions
    a["gap_same"] = np.log(a["close_same"]) - np.log(a["auction_price"])   # + = auction cheaper
    a["gap_prev"] = np.log(a["close_prev"]) - np.log(a["auction_price"])   # + = auction cheaper
    a["disc_prev"] = np.log(a["auction_price"]) - np.log(a["close_prev"])  # auction premium vs prev close
    a["ret_prev"] = np.log(a["close_prev"]).diff()

    print("=" * 78)
    print("SAMPLE")
    print(f"  auctions {len(a)}  {a['date'].min().date()} -> {a['date'].max().date()}")
    print(a.groupby(a["date"].dt.year).size().to_string())

    print("\n" + "=" * 78)
    print("GAP LEVEL (basis points of price)")
    for c in ("gap_same", "gap_prev", "disc_prev"):
        s = a[c] * 1e4
        print(f"  {c:<10} mean {s.mean():+7.2f}bp  sd {s.std():6.2f}  "
              f"q05 {s.quantile(.05):+7.2f}  q50 {s.median():+7.2f}  q95 {s.quantile(.95):+7.2f}  "
              f"P(>0) {np.mean(s > 0):.3f}")

    print("\n" + "=" * 78)
    print("PERSISTENCE")
    for c in ("gap_same", "gap_prev", "disc_prev"):
        s = a[c]
        ac = [s.autocorr(k) for k in (1, 2, 3, 5, 10, 22)]
        print(f"  {c:<10} ACF " + "  ".join(f"L{k}={v:+.3f}" for k, v in zip((1, 2, 3, 5, 10, 22), ac)))

    print("\n" + "=" * 78)
    print("EX-ANTE DRIVERS (known before the auction) vs gap_prev")
    a["logV"] = np.log(a["V"])
    a["days_since"] = a["date"].diff().dt.days
    cands = ["ret_prev", "logV", "days_since", "disc_prev"]
    for c in cands:
        x = a[c]
        ok = x.notna() & a["gap_prev"].notna()
        r = np.corrcoef(x[ok], a["gap_prev"][ok])[0, 1]
        rho = np.corrcoef(x[ok].rank(), a["gap_prev"][ok].rank())[0, 1]
        print(f"  corr(gap_prev, {c:<11}) = {r:+.3f}   rank {rho:+.3f}   n={ok.sum()}")

    print("\n" + "=" * 78)
    print("EX-POST DRIVERS (auction outcomes) vs gap_same")
    for c in ("R", "sd_B", "mu_B", "N", "S", "bid_price_spread_eur_per_tco2"):
        x = a[c].astype(float)
        ok = x.notna() & a["gap_same"].notna()
        rho = np.corrcoef(x[ok].rank(), a["gap_same"][ok].rank())[0, 1]
        print(f"  rank corr(gap_same, {c:<34}) = {rho:+.3f}")

    print("\n" + "=" * 78)
    print("NAIVE OUT-OF-SAMPLE BENCHMARKS for gap_prev (2023+ holdout)")
    cut = pd.Timestamp("2023-01-01")
    tr = a[a["date"] < cut]
    te = a[a["date"] >= cut]
    y = te["gap_prev"].values * 1e4
    preds = {
        "zero": np.zeros(len(te)),
        "train_mean": np.full(len(te), tr["gap_prev"].mean() * 1e4),
        "prev_gap (AR1)": te["gap_prev"].shift(1).values * 1e4,
        "prev_gap x train_rho": te["gap_prev"].shift(1).values * 1e4 * tr["gap_prev"].autocorr(1),
    }
    for name, p in preds.items():
        ok = ~np.isnan(p)
        mae = np.mean(np.abs(y[ok] - p[ok]))
        rmse = np.sqrt(np.mean((y[ok] - p[ok]) ** 2))
        sse = np.sum((y[ok] - p[ok]) ** 2)
        sst = np.sum((y[ok] - np.mean(y[ok])) ** 2)
        print(f"  {name:<22} n={ok.sum():>3}  MAE {mae:6.2f}bp  RMSE {rmse:6.2f}  R2 {1 - sse / sst:+.4f}")

    print("\n" + "=" * 78)
    print("ECONOMIC SCALE")
    a["eur_gap"] = a["close_same"] - a["auction_price"]
    print(f"  EUR/tCO2 gap: mean {a['eur_gap'].mean():+.3f}  sd {a['eur_gap'].std():.3f}")
    print(f"  value at stake per auction (gap x volume): mean EUR {np.mean(a['eur_gap'].abs() * a['V']):,.0f}")
    print(f"  test-window (2023+) total abs value at stake: EUR {np.sum((a['eur_gap'].abs() * a['V'])[a['date'] >= cut]):,.0f}")


if __name__ == "__main__":
    main()
