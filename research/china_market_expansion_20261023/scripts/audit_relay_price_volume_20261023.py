#!/usr/bin/env python3
"""Read-only quality screen for locally authorized SCEEX relay CSV archives.

Inputs are local-only row archives produced by fetch_sceex_snapshot.py.
Neither source rows nor corrected prices are written to public outputs.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


def screen(path):
    stats = collections.defaultdict(collections.Counter)
    issues = []
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with open(path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            code = row["instrument"]
            stats[code]["all_original_rows"] += 1
            try:
                p = float(row["mean_text"])
                v = float(row["volume_text"])
                a = float(row["turnover_text"])
            except (TypeError, ValueError):
                stats[code]["not_three_numeric_fields"] += 1
                continue
            stats[code]["three_numeric_fields"] += 1
            if not all(map(math.isfinite, (p, v, a))) or p <= 0 or v <= 0 or a < 0:
                stats[code]["nonpositive_or_nonfinite_triple"] += 1
                continue
            stats[code]["positive_finite_triples"] += 1
            implied = a / v
            tolerance = max(0.05, 0.001 * abs(implied))
            difference = abs(p - implied)
            if difference > tolerance:
                stats[code]["mean_vs_amount_over_volume_flag"] += 1
                issues.append({
                    "instrument": code,
                    "date": row["date"],
                    "source_page": row["source_page"],
                    "source_table_row": row["source_table_row"],
                    "absolute_difference_yuan_per_ton": round(difference, 6),
                    "source_page_sha256": row["source_sha256"],
                })
    fields = ["all_original_rows", "three_numeric_fields", "positive_finite_triples",
              "not_three_numeric_fields", "nonpositive_or_nonfinite_triple",
              "mean_vs_amount_over_volume_flag"]
    totals = {f: sum(c[f] for c in stats.values()) for f in fields}
    assert totals["all_original_rows"] == totals["three_numeric_fields"] + totals["not_three_numeric_fields"]
    assert totals["three_numeric_fields"] == totals["positive_finite_triples"] + totals["nonpositive_or_nonfinite_triple"]
    assert totals["mean_vs_amount_over_volume_flag"] == len(issues)
    return {
        "input_sha256": digest,
        "totals": totals,
        "by_instrument": {k: {f: stats[k][f] for f in fields} for k in sorted(stats)},
        "flag_locations": issues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                        help="Use NAME=absolute/local/path.csv; repeat for each snapshot.")
    args = parser.parse_args()
    output = {
        "audit_date": "2026-10-23",
        "classification": "READ_ONLY_FIELD_CONSISTENCY_DIAGNOSTIC",
        "not_official_semantics_certification": True,
        "changes_to_original_rows": 0,
        "model_fits": 0,
        "scored_origins": 0,
        "check": "abs(mean_text - turnover_text/volume_text) > max(0.05, 0.001*abs(turnover_text/volume_text))",
        "units_assumption": "Only a hypothetical yuan/ton amount-to-volume ratio; unit and trade channel must be certified by each exchange",
        "meaning": "Flag source-field disagreement; this DOES NOT establish an erroneous trade or license deletion/correction",
        "samples": {},
    }
    for item in args.input:
        if "=" not in item:
            parser.error("--input must use NAME=PATH")
        name, path = item.split("=", 1)
        output["samples"][name] = screen(path)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
