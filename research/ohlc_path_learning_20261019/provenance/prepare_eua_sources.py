#!/usr/bin/env python3
"""Acquire pinned author files and reconstruct two LOCAL, separate EUA series.

Python standard library only. No price series is bundled in this repository.
All numeric fields on Ding rows dated 2024 onward are excluded before numeric
conversion, validation, or descriptive statistics. Download/hash verification
necessarily handles the bytes of the complete, author-published source file.

Examples:
    python prepare_eua_sources.py --output-dir /tmp/carbon-path-local
    python prepare_eua_sources.py --output-dir /tmp/carbon-path-local --offline

For offline use, place the three files at output-dir/raw/{source_key}.csv,
or set --cache-dir to a directory with those file names. SHA-256 is always
verified. Existing files with a wrong hash fail closed and are not overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import urllib.request


SOURCES = {
    "quantum_x5": {
        "repository": "https://github.com/TravisCao/Quantum-LSTM",
        "commit": "048e63b8d59bd52c7eb9a031b5fb13268f2becb7",
        "path": "data/x_5d.csv",
        "sha256": "067fc8ebd2d05e1a4c702c5e5bc43f3bb76c6464909763bbe9dd1258e16270de",
        "bytes": 270861,
    },
    "quantum_y5": {
        "repository": "https://github.com/TravisCao/Quantum-LSTM",
        "commit": "048e63b8d59bd52c7eb9a031b5fb13268f2becb7",
        "path": "data/y_5d.csv",
        "sha256": "9742bfbe1093e02c531ca659ee894796719462d9a401f577ec02c156975265ca",
        "bytes": 9572,
    },
    "ding_matrix": {
        "repository": "https://github.com/dinggaoliu/Interval-valued-carbon-prices-forecasting",
        "commit": "25097e0a700e56573a12c6657c5df92466c508ab",
        "path": "original_data_matrix.csv",
        "sha256": "a955e26901642b8ff7d68e47988228f773921cbe064b6e722b4e95d224e244c0",
        "bytes": 67735,
    },
}
OUT_FIELDS = ["source_index", "date", "price", "open", "high", "low", "volume"]
Q_FIELDS = ["Price", "Open", "High", "Low", "Vol", "Week"]
DING_CUTOFF = "2024-01-01"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_non_git_path(path: Path) -> Path:
    """Make accidental publication of a reconstructed series harder."""
    resolved = path.expanduser().resolve()
    for directory in (resolved, *resolved.parents):
        marker = directory / ".git"
        real_git_directory = marker.is_dir() and (marker / "HEAD").is_file()
        git_worktree_file = marker.is_file() and marker.read_text(encoding="utf-8").startswith("gitdir:")
        if real_git_directory or git_worktree_file:
            raise ValueError(f"Raw/derived prices must stay outside a Git worktree: {resolved}")
    return resolved


def acquire(cache: Path, offline: bool) -> tuple[dict, list]:
    cache.mkdir(parents=True, exist_ok=True)
    paths, receipt = {}, []
    for key, source in SOURCES.items():
        repository_path = source["repository"].split("github.com/", 1)[1]
        url = "https://raw.githubusercontent.com/{}/{}/{}".format(
            repository_path, source["commit"], source["path"]
        )
        path = cache / f"{key}.csv"
        record = {"source_key": key, **source, "url": url, "cached": path.exists()}
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"Offline source missing: {path}")
            record["download_started_utc"] = datetime.now(timezone.utc).isoformat()
            with urllib.request.urlopen(url, timeout=60) as response:
                if response.status != 200:
                    raise RuntimeError(f"Source GET returned HTTP {response.status}: {url}")
                raw = response.read()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != source["sha256"] or len(raw) != source["bytes"]:
                raise ValueError(f"Downloaded source does not match pinned bytes: {key}")
            temporary = path.with_suffix(".download")
            temporary.write_bytes(raw)
            temporary.replace(path)
        if sha256(path) != source["sha256"] or path.stat().st_size != source["bytes"]:
            raise ValueError(f"Cached source does not match pinned bytes: {key}")
        record["verified"] = True
        paths[key] = path
        receipt.append(record)
    return paths, receipt


def decimal_value(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError("Invalid numeric input in an admitted source row") from error
    if not value.is_finite():
        raise ValueError("Non-finite numeric input in an admitted source row")
    return value


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def quantum_rows(x_path: Path, y_path: Path) -> tuple[list, dict]:
    x, y = read_csv(x_path), read_csv(y_path)
    required = [f"{field}-{lag}" for lag in range(1, 6) for field in Q_FIELDS]
    if not x or list(x[0]) != required or not y or list(y[0]) != ["Price"]:
        raise ValueError("Unexpected Quantum source schema")
    if len(x) != 1801 or len(y) != len(x):
        raise ValueError("Unexpected Quantum window/label count")

    # Original rows run newest to oldest. Each next raw row moves back one day.
    # Exact decimal equality checks all six fields, including the weekday hint.
    overlap_checks = 0
    for newer, older in zip(x, x[1:]):
        for lag in range(1, 5):
            for field in Q_FIELDS:
                if decimal_value(newer[f"{field}-{lag + 1}"]) != decimal_value(older[f"{field}-{lag}"]):
                    raise ValueError("Quantum adjacent window overlap does not match")
                overlap_checks += 1
    label_checks = 0
    for row, label in zip(x[:-1], y[1:]):
        if decimal_value(row["Price-1"]) != decimal_value(label["Price"]):
            raise ValueError("Quantum next-price labels do not align with windows")
        label_checks += 1

    def take(row: dict, lag: int) -> dict:
        return {
            "date": "",  # Calendar dates are not in the supplied window files.
            "price": row[f"Price-{lag}"],
            "open": row[f"Open-{lag}"],
            "high": row[f"High-{lag}"],
            "low": row[f"Low-{lag}"],
            "volume": row[f"Vol-{lag}"],
        }

    oldest_to_newest = list(reversed(x))
    rows = [take(oldest_to_newest[0], lag) for lag in (5, 4, 3, 2, 1)]
    rows.extend(take(row, 1) for row in oldest_to_newest[1:])
    if len(rows) != 1805:
        raise ValueError("Unexpected reconstructed Quantum sequence length")
    for index, row in enumerate(rows):
        row["source_index"] = index

    # y[0] is the next price after the final complete row. Its O/H/L/V do not
    # exist in x_5d, so it is deliberately NOT appended as a complete record.
    audit = {
        "source_window_rows": len(x), "reconstructed_rows": len(rows),
        "window_overlap_scalar_checks": overlap_checks,
        "window_overlap_errors": 0,
        "adjacent_price_label_checks": label_checks,
        "adjacent_price_label_errors": 0,
        "calendar_dates_available": False,
        "ordering": "Oldest-to-newest sequence inferred from lag and next-label structure",
        "author_described_period": "2014-01-01 to 2020-12-31; not a recovered daily calendar",
        "unpaired_final_price_label_appended": False,
        "price_semantics": "Author column Price; authentic traded close/settlement/roll unresolved",
        "volume_units": "Author Vol numeric units; no unit conversion or cross-source comparison",
    }
    return rows, audit


def ding_rows(path: Path) -> tuple[list, dict]:
    rows = []
    excluded_date_rows = 0
    last_admitted_date = None
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["Date", "high", "low", "close", "open", "volumek", "gas", "oil", "coal", "power", "stock", "pos", "neg", "trends"]:
            raise ValueError("Unexpected Ding source schema")
        for source_index, source in enumerate(reader):
            # Date guard must precede all numeric field access/conversion.
            date = datetime.strptime(source["Date"], "%Y/%m/%d").strftime("%Y-%m-%d")
            if date >= DING_CUTOFF:
                excluded_date_rows += 1
                continue
            if date < "2021-01-01":
                raise ValueError("Unexpected early Ding source date")
            if last_admitted_date is not None and date <= last_admitted_date:
                raise ValueError("Ding admitted dates are not strictly increasing and unique")
            last_admitted_date = date
            rows.append({
                "source_index": source_index, "date": date,
                "price": source["close"], "open": source["open"],
                "high": source["high"], "low": source["low"],
                "volume": source["volumek"],
            })
    if len(rows) != 772:
        raise ValueError("Unexpected Ding pre-2024 admitted row count")
    return rows, {
        "reconstructed_rows": len(rows), "date_min": rows[0]["date"],
        "date_max": rows[-1]["date"], "exclusive_date_cutoff": DING_CUTOFF,
        "excluded_rows_by_date_only": excluded_date_rows,
        "numeric_values_of_excluded_rows_evaluated": False,
        "price_semantics": "Author column close; exact contract, last-trade/settlement, and roll unresolved",
        "volume_units": "Author volumek units retained; no conversion or cross-source comparison",
        "external_features_used": False,
    }


def descriptive_audit(rows: list) -> dict:
    numeric = [{key: decimal_value(row[key]) for key in OUT_FIELDS[2:]} for row in rows]
    return {
        "row_count": len(rows),
        "nonpositive_price_ohl_count": sum(min(r[k] for k in ("price", "open", "high", "low")) <= 0 for r in numeric),
        "negative_volume_count": sum(r["volume"] < 0 for r in numeric),
        "high_below_low_count": sum(r["high"] < r["low"] for r in numeric),
        "open_outside_high_low_count": sum(not r["low"] <= r["open"] <= r["high"] for r in numeric),
        "price_outside_high_low_count": sum(not r["low"] <= r["price"] <= r["high"] for r in numeric),
        "zero_range_count": sum(r["high"] == r["low"] for r in numeric),
        "zero_volume_count": sum(r["volume"] == 0 for r in numeric),
        "containment_repairs_applied": False,
        "interpolation_or_imputation_applied": False,
    }


def save_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/carbon-path-local"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    output = require_non_git_path(args.output_dir)
    cache = require_non_git_path(args.cache_dir or output / "raw")
    output.mkdir(parents=True, exist_ok=True)
    paths, receipt = acquire(cache, args.offline)
    quantum, q_audit = quantum_rows(paths["quantum_x5"], paths["quantum_y5"])
    ding, d_audit = ding_rows(paths["ding_matrix"])
    audits = {}
    for name, rows, audit in (("quantum", quantum, q_audit), ("ding", ding, d_audit)):
        save_csv(output / f"{name}.csv", rows)
        audits[name] = {
            **audit, **descriptive_audit(rows),
            "output_sha256": sha256(output / f"{name}.csv"),
            "output_bytes": (output / f"{name}.csv").stat().st_size,
            "output_fields": OUT_FIELDS,
        }
    audit = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "separate_author_sources_same_EUA_market": True,
        "sources_merged": False,
        "warning": "O/H/L and Price semantics are not authenticated; do not force Price inside H/L.",
        "datasets": audits, "source_acquisition": receipt,
    }
    (output / "LOCAL_DATA_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
