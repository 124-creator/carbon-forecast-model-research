"""Add only origin-known micro scale and future observable auxiliary labels.

Input is the corrected development study. Original target identities and all
original fields are retained. No final supply or future median enters inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(parent, output):
    raw = np.load(parent/"local_only/examples_development.npz")
    arrays = {k: raw[k].copy() for k in raw.files}
    meta = pd.read_csv(parent/"local_only/example_metadata.csv")
    ev = pd.read_csv(parent/"local_only/auction_events_development.csv")
    ev = ev.sort_values(["date", "platform"], kind="stable")
    if meta.target_date.max() >= "2024-01-01" or ev.date.max() >= "2024-01-01":
        raise ValueError("Development-only builder")
    lookup = ev.set_index("event_id", verify_integrity=True)
    medcol, pcol = "bid_median_eur_per_tco2", "clearing_price_eur_per_tco2"
    med0 = arrays["last_bid_median"].copy()
    if not np.all(np.isfinite(med0) & (med0 > 0)):
        raise ValueError("This frozen study requires an observed positive reference median")
    cache, scales, last_dates, counts = {}, [], [], []
    for row in meta.itertuples(index=False):
        if row.origin_date not in cache:
            hist = ev[(ev.date <= row.origin_date) & np.isfinite(ev[pcol]) & np.isfinite(ev[medcol])
                      & (ev[pcol] > 0) & (ev[medcol] > 0)].tail(60)
            if len(hist) < 60:
                raise ValueError("Insufficient historical micro-scale observations")
            gap = np.abs(hist[pcol].to_numpy()-hist[medcol].to_numpy())
            ref_median = float(lookup.loc[row.reference_event_id, medcol])
            micro = max(float(np.median(gap)), .01, .0001*ref_median)
            cache[row.origin_date] = (micro, hist.date.max(), len(hist))
        scale, date, count = cache[row.origin_date]
        scales.append(scale)
        last_dates.append(date)
        counts.append(count)
    micro = np.asarray(scales, dtype=np.float64)
    future_median = np.full(len(meta), np.nan, dtype=np.float64)
    for i, event_id in enumerate(meta.result_event_id):
        if isinstance(event_id, str) and event_id in lookup.index:
            future_median[i] = float(lookup.loc[event_id, medcol])
    has_median = np.isfinite(future_median) & (future_median > 0)
    has_basis = has_median & arrays["completed"].astype(bool)
    med_ratio = np.where(has_median, future_median/med0, np.nan)
    median_z = np.where(has_median, (future_median-med0)/arrays["scale"], np.nan)
    eps = micro/med0
    basis_z = np.where(has_basis,
                       (arrays["price"]/future_median-arrays["anchor"]/med0)/eps, np.nan)
    old_base = arrays["base"].astype(np.float64)
    # Recover exact physical ratios from raw origin fields / stored scale.
    b0 = np.array([float(lookup.loc[i, "total_bid_volume_tco2"]) for i in meta.reference_event_id])
    s0 = np.array([float(lookup.loc[i, "supply_tco2"]) for i in meta.reference_event_id])
    cal = pd.read_csv(parent/"local_only/calendar_vintages_development.csv")
    calmap = cal.set_index(["vintage_id", "scheduleddate", "platform"])["plannedvolume"]
    planned = np.array([float(calmap.loc[(r.calendar_vintage_id, r.target_date, r.platform)])
                        for r in meta.itertuples(index=False)])
    arrays["base"] = np.column_stack([s0/b0, planned/b0, arrays["scale"]/arrays["anchor"],
                                      eps, med0/arrays["anchor"]]).astype(np.float64)
    arrays["context"] = np.column_stack([arrays["context"], np.log(micro/arrays["anchor"]),
                                         np.log(med0/arrays["anchor"])]).astype(np.float32)
    arrays.update(median_z=median_z, median_ratio=med_ratio, has_median=has_median,
                  basis_z=basis_z, has_basis=has_basis, micro_scale=micro,
                  future_median_label=future_median)
    assert np.all(np.array(last_dates) < meta.issue_date.to_numpy())
    assert arrays["base"].shape == (len(meta), 5)
    private = output/"local_only"
    private.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(private/"examples_development.npz", **arrays)
    meta.to_csv(private/"example_metadata.csv", index=False)
    audit = json.loads((parent/"EXAMPLE_AUDIT.json").read_text())
    audit["parent_examples_sha256"] = sha(parent/"local_only/examples_development.npz")
    audit["examples_sha256"] = sha(private/"examples_development.npz")
    audit["metadata_sha256"] = sha(private/"example_metadata.csv")
    audit["context_names"] += ["log_micro_scale_over_price", "log_last_median_over_price"]
    audit["factorisation"] = {
        "micro_scale": "median of absolute reported clearing-minus-median difference over last 60 observed auctions; floor max(0.01 EUR, 0.0001*reference median); no horizon multiplier",
        "base_dtype": "float64; physical ratios rebuilt from the same raw known origin quantities and corrected as-of calendar",
        "unchanged_original_base_max_rounding_difference": float(np.max(np.abs(arrays["base"][:, :3]-old_base))),
        "future_median_role": "label only; never in x, context, base, normalization fit or inference",
        "basis_label": "(future price / future median - current price / current median) / (micro_scale/current median)",
        "n_median_labels": int(has_median.sum()), "n_basis_labels": int(has_basis.sum()),
        "minimum_history_count": int(min(counts)), "history_timestamp_check": "PASS",
        "n_examples_unchanged": len(meta), "origin_known_micro_scale_quantiles": np.quantile(micro, [0, .25, .5, .75, 1]).tolist()
    }
    (output/"EXAMPLE_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit["factorisation"], ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-study", required=True, type=Path)
    ap.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    build(args.source_study, args.study_dir)
