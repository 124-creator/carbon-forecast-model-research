"""Construct examples from past results and complete as-of calendar vintages.

Forecasts are issued at 00:00 UTC on the day after an observed auction day.
Only the previous day's and earlier results are inputs. Target identities are
the h-th future scheduled (date, platform) event in the currently known plan.
They are selected before looking up outcomes, including cancellations/missing
results. Annual final results are not themselves a complete publication archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLS = ["clearing_price_eur_per_tco2", "bid_min_eur_per_tco2",
              "bid_max_eur_per_tco2", "bid_mean_eur_per_tco2", "bid_median_eur_per_tco2"]
VOLUME_COLS = ["supply_tco2", "total_bid_volume_tco2", "average_bid_size_tco2",
               "average_bid_volume_per_bidder_tco2", "bid_volume_std_per_bidder_tco2"]
COUNT_COLS = ["bid_count", "bidder_count", "successful_bid_count", "successful_bidder_count"]
PLATFORMS = ["CAP", "DE", "PL", "NI"]
FEATURE_NAMES = (["relative_"+c for c in PRICE_COLS] +
                 ["observed_"+c for c in PRICE_COLS] +
                 ["log_"+c for c in VOLUME_COLS] +
                 ["observed_"+c for c in VOLUME_COLS] +
                 ["log_"+c for c in COUNT_COLS] +
                 ["observed_"+c for c in COUNT_COLS] +
                 ["log_cover", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
                  "log_gap_days", "price_missing"] + ["platform_"+p for p in PLATFORMS])
CONTEXT_NAMES = ["log_plannedS_over_B0", "log_plannedS_over_S0", "log_days_to_target",
                 "log_horizon", "target_month_sin", "target_month_cos", "target_weekday_sin",
                 "target_weekday_cos", "indicative_plan", "log_vintage_age_days",
                 "log_current_price", "log_current_volatility"] + ["target_platform_"+p for p in PLATFORMS]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def known_plan(cal, issue, history_date):
    eligible = cal[cal.revision_available_date <= issue]
    plans = []
    for year, group in eligible.groupby("calendar_year", sort=True):
        newest = group.revision_available_date.max()
        ids = group.loc[group.revision_available_date == newest, "vintage_id"].unique()
        if len(ids) != 1:
            raise ValueError(f"Unresolved contemporaneous calendar vintages: {year} {newest}")
        vintage = group[group.vintage_id == ids[0]]
        expiry = vintage.valid_until_exclusive.iloc[0]
        if pd.notna(expiry) and issue >= expiry:
            continue
        plans.append(vintage[vintage.scheduleddate > history_date])
    if not plans:
        return cal.iloc[:0]
    return pd.concat(plans).sort_values(["scheduleddate", "platform"], kind="stable")


def make_history(hist, anchor, vol):
    price = hist[PRICE_COLS].to_numpy(float)
    pmask = np.isfinite(price)
    pr = np.clip((np.nan_to_num(price, nan=anchor)-anchor)/vol, -40, 40)
    vv = hist[VOLUME_COLS].to_numpy(float)
    vm = np.isfinite(vv) & (vv > 0)
    lv = np.log(np.where(vm, vv/1e6, 1)).clip(-12, 12)
    cc = hist[COUNT_COLS].to_numpy(float)
    cm = np.isfinite(cc) & (cc >= 0)
    lc = np.log1p(np.where(cm, cc, 0))
    cover = (hist.total_bid_volume_tco2/hist.supply_tco2).to_numpy(float)
    cover = np.log(np.nan_to_num(cover, nan=1).clip(.01, 100))
    dates = hist.date
    wd, month = dates.dt.dayofweek.to_numpy(), dates.dt.month.to_numpy()
    gaps = dates.diff().dt.total_seconds().div(86400).fillna(0).to_numpy()
    misc = np.column_stack([cover, np.sin(2*np.pi*wd/7), np.cos(2*np.pi*wd/7),
                            np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
                            np.log1p(gaps), 1-pmask[:, 0]])
    platform = np.column_stack([(hist.platform == p).to_numpy() for p in PLATFORMS])
    x = np.column_stack([pr, pmask, lv, vm, lc, cm, misc, platform]).astype(np.float32)
    assert x.shape == (20, len(FEATURE_NAMES)) and np.isfinite(x).all()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    args = ap.parse_args()
    root = args.study_dir
    event_path = root/"local_only/auction_events_development.csv"
    calendar_path = root/"local_only/calendar_vintages_development.csv"
    ev = pd.read_csv(event_path, parse_dates=["date"]).sort_values(["date", "platform"], kind="stable").reset_index(drop=True)
    cal = pd.read_csv(calendar_path, parse_dates=["scheduleddate", "revision_available_date", "valid_until_exclusive"])
    if ev.date.max().year > 2023 or cal.scheduleddate.max().year > 2023:
        raise ValueError("Development builder refuses confirmation years")
    if ev.duplicated(["date", "platform"]).any():
        raise ValueError("Source identity is not unique under chosen scheduled target key")
    lookup = {(r.date, r.platform): r for r in ev.itertuples(index=False)}
    arrays = {k: [] for k in ["x", "context", "base", "y", "price", "anchor", "scale", "bid_ratio", "completed", "has_bid", "known_status", "cancelled", "last_bid_median"]}
    metadata, skipped = [], Counter()
    for origin in ev.date.drop_duplicates():
        issue = origin + pd.Timedelta(days=1)
        if issue.year < 2017:
            continue
        past = ev[ev.date <= origin]
        if len(past) < 20:
            skipped["history_short"] += 1
            continue
        valid = past[np.isfinite(past.clearing_price_eur_per_tco2) & (past.clearing_price_eur_per_tco2 > 0)]
        if len(valid) < 61:
            skipped["volatility_history_short"] += 1
            continue
        ref = valid.iloc[-1]
        p0, b0, s0 = float(ref.clearing_price_eur_per_tco2), float(ref.total_bid_volume_tco2), float(ref.supply_tco2)
        if not (b0 > 0 and s0 > 0):
            skipped["reference_volume_absent"] += 1
            continue
        recent = valid.iloc[-61:]
        returns = np.diff(np.log(recent.clearing_price_eur_per_tco2.to_numpy(float)))
        vol = max(float(np.std(returns, ddof=1))*p0, .002*p0, .01)
        mean_gap = max(float(recent.date.diff().dt.days.dropna().clip(lower=1).mean()), 1.)
        hist = past.iloc[-20:]
        x = make_history(hist, p0, vol)
        plan = known_plan(cal, issue, origin)
        for horizon in args.horizons:
            if len(plan) < horizon:
                skipped[f"no_known_target_h{horizon}"] += 1
                continue
            target = plan.iloc[horizon-1]
            targetdate = target.scheduleddate
            if targetdate.year > 2023:
                raise AssertionError("Confirmation target accessed")
            gap = (targetdate-origin).days
            scale = vol*np.sqrt(max(gap/mean_gap, 1.))
            supply = float(target.plannedvolume)
            age = (issue-target.revision_available_date).days
            indicative = str(target.volume_is_indicative).lower() == "true"
            context = np.array([np.log(supply/b0), np.log(supply/s0), np.log1p(gap), np.log(horizon),
                                np.sin(2*np.pi*targetdate.month/12), np.cos(2*np.pi*targetdate.month/12),
                                np.sin(2*np.pi*targetdate.dayofweek/7), np.cos(2*np.pi*targetdate.dayofweek/7),
                                indicative, np.log1p(age), np.log(p0), np.log(vol/p0)] +
                               [target.platform == p for p in PLATFORMS], dtype=np.float32)
            # The target has now been fixed WITHOUT looking at the result table.
            outcome = lookup.get((targetdate, target.platform))
            price = float(outcome.clearing_price_eur_per_tco2) if outcome is not None else np.nan
            bid = float(outcome.total_bid_volume_tco2) if outcome is not None else np.nan
            completed = np.isfinite(price) and price > 0
            has_bid = np.isfinite(bid) and bid > 0
            cancelled = outcome is not None and outcome.event_status == "reported_cancelled_or_failed"
            known_status = completed or cancelled
            row = {"x": x, "context": context, "base": [s0/b0, supply/b0, scale/p0],
                   "y": (price-p0)/scale if completed else np.nan,
                   "price": price, "anchor": p0, "scale": scale,
                   "bid_ratio": bid/b0 if has_bid else np.nan, "completed": completed,
                   "has_bid": has_bid, "known_status": known_status, "cancelled": cancelled,
                   "last_bid_median": float(ref.bid_median_eur_per_tco2)}
            for key, value in row.items():
                arrays[key].append(value)
            metadata.append({"example_id": f"{issue.date()}_h{horizon}_{targetdate.date()}_{target.platform}",
                             "origin_date": str(origin.date()), "issue_date": str(issue.date()),
                             "target_date": str(targetdate.date()), "issue_year": issue.year,
                             "horizon": horizon, "platform": target.platform,
                             "history_last_date": str(hist.date.max().date()),
                             "reference_event_id": ref.event_id,
                             "calendar_vintage_id": target.vintage_id,
                             "calendar_available_date": str(target.revision_available_date.date()),
                             "calendar_valid_until_exclusive": str(target.valid_until_exclusive.date()) if pd.notna(target.valid_until_exclusive) else "",
                             "calendar_indicative": indicative,
                             "result_event_id": outcome.event_id if outcome is not None else "",
                             "result_found": outcome is not None, "completed": completed,
                             "cancelled": cancelled,
                             "supply_plan_matches_final": bool(np.isclose(supply, outcome.supply_tco2, rtol=0, atol=.01)) if outcome is not None else None})
    meta = pd.DataFrame(metadata)
    if len(meta) == 0:
        raise ValueError("No causally eligible examples")
    if meta.example_id.duplicated().any():
        raise ValueError("Duplicate issued example")
    assert (meta.history_last_date < meta.issue_date).all()
    assert (meta.calendar_available_date <= meta.issue_date).all()
    assert (meta.target_date >= meta.issue_date).all()
    precise = {"y", "price", "anchor", "scale", "bid_ratio", "last_bid_median"}
    np.savez_compressed(root/"local_only/examples_development.npz",
                        **{k: np.asarray(v, dtype=np.float64 if k in precise else np.float32)
                           for k, v in arrays.items()})
    meta.to_csv(root/"local_only/example_metadata.csv", index=False)
    # Price-only histories deliberately keep missing/time/platform information;
    # calendar context is held fixed to isolate the new auction-result inputs.
    price_only_columns = [i for i, name in enumerate(FEATURE_NAMES)
                          if name in {"relative_clearing_price_eur_per_tco2", "observed_clearing_price_eur_per_tco2",
                                      "weekday_sin", "weekday_cos", "month_sin", "month_cos", "log_gap_days", "price_missing"}
                          or name.startswith("platform_")]
    audit = {"event_sha256": sha(event_path), "calendar_sha256": sha(calendar_path),
             "examples_sha256": sha(root/"local_only/examples_development.npz"),
             "metadata_sha256": sha(root/"local_only/example_metadata.csv"),
             "n_examples": len(meta), "n_origins": meta.issue_date.nunique(),
             "counts_by_year_horizon": meta.groupby(["issue_year", "horizon"]).agg(
                 issued=("example_id", "size"), completed=("completed", "sum"),
                 cancelled=("cancelled", "sum"), found=("result_found", "sum")).reset_index().to_dict("records"),
             "skipped_origin_horizon_counts": dict(skipped), "feature_names": FEATURE_NAMES,
             "context_names": CONTEXT_NAMES, "price_only_columns": price_only_columns,
             "causal_timestamp_assertions": "PASS", "target_selection": "complete known annual vintage; then scheduled date/platform rank; then outcome lookup",
             "horizon_units": "scheduled auction events, NOT trading days",
             "result_availability_assumption": "Annual final event fields treated as known by 00:00 UTC on following date; no full historical intraday result publication archive was obtained",
             "calendar_availability": "explicit document publication date plus one natural day; conflicting/missing versions invalidate old plans",
             "missing_target_result_count": int((~meta.result_found).sum()),
             "supply_plan_final_mismatch_count": int((meta.supply_plan_matches_final == False).sum())}
    (root/"EXAMPLE_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_examples": len(meta), "n_origins": audit["n_origins"],
                      "counts": audit["counts_by_year_horizon"], "checks": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
