#!/usr/bin/env python3
"""Read-only 2018-2024 HBEA/GDEA relay-field screening.

No source row is deleted, corrected, interpolated or scored. This is NOT
first-party channel/price certification and is NOT a forecasting experiment.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

FROM = "2018-01-01"
TO = "2024-12-31"
ALLOWED = ("HBEA", "GDEA")


def finite(text):
    try:
        value = float(text)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def period(code, date):
    if code != "GDEA":
        return "no_predeclared_date_cut"
    if date < "2018-07-31":
        return "pre_2018_07_31"
    if date < "2019-01-07":
        return "2018_07_31_to_2019_01_06"
    return "from_2019_01_07"


def tally(rows, expected, include_issues=True):
    c = collections.Counter()
    issues = []
    dates = collections.Counter()
    for r in rows:
        c["source_rows"] += 1
        code_ok = (r["instrument"] == expected)
        c["expected_code_rows"] += code_ok
        if not code_ok:
            c["unexpected_code_rows"] += 1
            if include_issues:
                issues.append({
                "kind": "instrument_code_mismatch",
                "date": r["date"],
                "observed_code": r["instrument"],
                "source_file": r["source_file"],
                "source_table_row": r["source_table_row"],
            })
        dates[r["date"]] += 1
        mean = finite(r["mean_text"])
        close = finite(r["close_text"])
        volume = finite(r["volume_text"])
        amount = finite(r["turnover_text"])
        c["mean_numeric"] += mean is not None
        c["close_numeric"] += close is not None
        c["positive_close"] += close is not None and close > 0
        c["volume_numeric"] += volume is not None
        c["positive_volume"] += volume is not None and volume > 0
        c["positive_close_missing_or_nonpositive_volume"] += (
            close is not None and close > 0 and (volume is None or volume <= 0)
        )
        valid = (mean is not None and mean > 0 and
                 volume is not None and volume > 0 and
                 amount is not None and amount >= 0)
        if not valid:
            c["not_positive_numeric_triplet"] += 1
            continue
        c["positive_numeric_triplet"] += 1
        c["triplet_and_expected_code"] += code_ok
        implied = amount / volume
        delta = abs(mean - implied)
        if delta > max(0.05, 0.001 * abs(implied)):
            c["field_ratio_disagreement_flags"] += 1
            if include_issues:
                issues.append({
                "kind": "mean_vs_amount_div_volume",
                "date": r["date"],
                "source_file": r["source_file"],
                "source_table_row": r["source_table_row"],
                "expected_instrument_code": code_ok,
                "absolute_numeric_difference": round(delta, 5),
            })
    duplicates = [
        {"date": day, "source_rows_same_date": num}
        for day, num in sorted(dates.items()) if num > 1
    ]
    out = {
        "counts": {
            field: c[field] for field in (
                "source_rows", "expected_code_rows", "unexpected_code_rows",
                "mean_numeric", "close_numeric", "positive_close",
                "volume_numeric", "positive_volume", "positive_numeric_triplet",
                "triplet_and_expected_code", "not_positive_numeric_triplet",
                "positive_close_missing_or_nonpositive_volume",
                "field_ratio_disagreement_flags"
            )
        },
        "distinct_source_dates": len(dates),
        "duplicate_date_groups": duplicates,
        "issue_locations_and_derived_magnitudes_only": issues,
    }
    assert c["source_rows"] == c["positive_numeric_triplet"] + c["not_positive_numeric_triplet"]
    assert c["source_rows"] == c["expected_code_rows"] + c["unexpected_code_rows"]
    return out


def read_market(path, code):
    raw = Path(path).read_bytes()
    data = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    assert data and all("date" in r for r in data)
    filtered = [r for r in data if FROM <= r["date"] <= TO]
    yearly = {
        str(year): tally([r for r in filtered if r["date"].startswith(str(year))], code, include_issues=False)
        for year in range(2018, 2025)
    }
    by_regime = {}
    if code == "GDEA":
        for key in ("pre_2018_07_31", "2018_07_31_to_2019_01_06", "from_2019_01_07"):
            by_regime[key] = tally([r for r in filtered if period(code, r["date"]) == key], code, include_issues=False)
    overall = tally(filtered, code)
    assert overall["counts"]["source_rows"] == sum(
        s["counts"]["source_rows"] for s in yearly.values())
    return {
        "source_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows_in_fixed_period": len(filtered),
        "overall": overall,
        "by_year": yearly,
        "guangdong_candidate_rule_regimes": by_regime,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbea-file", required=True)
    ap.add_argument("--gdea-file", required=True)
    args = ap.parse_args()
    result = {
        "audit_id": "CHINA-HBEA-GDEA-FIELD-SEMANTICS-20261023",
        "fixed_period": {"from": FROM, "through": TO},
        "scope": "Data quality only; zero model fits, scoring origins or QLIKE comparisons",
        "diagnostic": "abs(mean - amount/volume) > max(0.05, 0.001*abs(amount/volume))",
        "meaning": "A flag may result from differing trade channels, definitions or transcription; it is not proof of bad data",
        "guangdong_rule_dates_source_status": "2018-07-31 and 2019-01-07 from contemporaneous exchange-authored notices reproduced by independent secondary outlet; original exchange URL not retrieved",
        "original_rows_retained": True,
        "forward_filled": False,
        "2025_or_2026_confirmatory_use": False,
        "market_results": {
            "HBEA": read_market(args.hbea_file, "HBEA"),
            "GDEA": read_market(args.gdea_file, "GDEA"),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
