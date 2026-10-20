"""Prepare the fixed EEX archive locally under pre-price scope and addendum.

No network, forecasting, fitting, data distribution, interpolation or price
summary is performed. Output is a new private directory; existing paths fail.
Requires xlrd 2.0.1 and openpyxl 3.1.5. Source ZIP preserves exact Excel bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re

from inspect_eex_source import (
    ARCHIVE_SHA256, SCOPE_SHA256, SHEET, as_decimal, digest, norm,
    serializable, source_books, source_rows, utc,
)

ADDENDUM_SHA256 = "13368f9e7dc0d33ae68d5d12c5611c7e4336a96c06f61656158fe43d7eb23e97"
INCLUDED_CONTRACT = "T3PA"
KNOWN_CONTRACTS = {"T3PA", "T2PA", "EAA2", "EAA3"}
DEV_END = "2024-01-01"
CONFIRM_END = "2026-01-01"
FAILURE_PATTERN = re.compile(r"cancel|unsuccess|not successful|not held|failed|failure", re.I)
DAILY_FIELDS = [
    "date", "price", "auction_volume", "total_bid_volume", "cover_ratio",
    "bidder_count", "successful_bidder_count", "auction_count", "contract",
    "auction_names", "completion_evidence", "missing_fields", "component_ids",
]
COMPONENT_FIELDS = [
    "date", "time", "auction_name", "contract", "price", "auction_volume",
    "total_bid_volume", "reported_cover_ratio", "bidder_count",
    "successful_bidder_count", "revenue", "source_status", "completion_evidence",
    "source_year", "source_member", "source_sheet", "source_excel_row", "component_id",
]


def decimal_text(number):
    if number is None:
        return ""
    if not number.is_finite():
        raise ValueError("Nonfinite output forbidden")
    return format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else format(number, "f")


def time_text(value):
    if isinstance(value, dt.datetime):
        return value.time().isoformat()
    if isinstance(value, dt.time):
        return value.isoformat()
    if isinstance(value, str) and re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", value.strip()):
        return value.strip()
    raise ValueError("Auction time format is not authenticated")


def identity(row):
    return (row["date"], time_text(row["time"]), str(row["auction_name"]).strip(), str(row["contract"]).strip())


def validate_completed(row, has_status):
    """Return one evidence label or an exclusion category; never guess a status."""
    contract = str(row["contract"]).strip()
    if contract not in KNOWN_CONTRACTS:
        raise ValueError("Unexpected product code; source definition review required")
    if not ("2012-01-01" <= row["date"] < CONFIRM_END):
        raise ValueError("Source date outside frozen archive range")
    if contract != INCLUDED_CONTRACT:
        return None, "excluded_contract"
    status = norm(row["status"])
    has_failure = any(isinstance(v, str) and FAILURE_PATTERN.search(v) for v in row["source_cells"])
    if has_status:
        if status == "cancelled":
            return None, "explicit_cancelled"
        if status != "successful":
            raise ValueError("Unknown/empty final status in modern report")
        evidence = "explicit_successful"
    else:
        if row["source_year"] > 2019:
            raise ValueError("Missing status in modern report")
        evidence = "inferred_completed"
    if has_failure:
        return None, "failure_marker"
    price, volume, revenue = [as_decimal(row[k]) for k in ("price", "volume", "revenue")]
    if any(x is None or x <= 0 for x in (price, volume, revenue)):
        return None, "missing_or_nonpositive_required_value"
    with localcontext() as context:
        context.prec = 50
        if abs(revenue - price * volume) > Decimal("0.01"):
            return None, "revenue_identity_failure"
    if row["certificate"] is not None and str(row["certificate"]).strip() != "EUA":
        raise ValueError("Certificate conflicts with approved EUA contract")
    return evidence, None


def optional_nonnegative(value):
    number = as_decimal(value)
    if number is not None and number < 0:
        raise ValueError("Negative reported optional count/volume/ratio; source review required")
    return number


def make_component(row, evidence):
    ident = identity(row)
    component_id = "eex_" + hashlib.sha256(json.dumps(ident, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:24]
    return {
        "date": row["date"], "time": ident[1], "auction_name": ident[2], "contract": ident[3],
        "price": as_decimal(row["price"]), "auction_volume": as_decimal(row["volume"]),
        "total_bid_volume": optional_nonnegative(row["total_bids"]),
        "reported_cover_ratio": optional_nonnegative(row["cover_ratio"]),
        "bidder_count": optional_nonnegative(row["total_bidders"]),
        "successful_bidder_count": optional_nonnegative(row["successful_bidders"]),
        "revenue": as_decimal(row["revenue"]), "source_status": norm(row["status"]) or "column_absent",
        "completion_evidence": evidence, "source_year": row["source_year"],
        "source_member": row["source_member"], "source_sheet": SHEET,
        "source_excel_row": row["excel_row"], "component_id": component_id,
    }


def deduplicate(rows):
    """Exact official-identity duplicates only; conflicting source rows fail."""
    seen, output, duplicate_locations = {}, [], []
    for row in rows:
        ident = identity(row)
        # Location metadata may differ; official numeric/text result content may not.
        payload = json.dumps({k: serializable(v) for k, v in row.items()
                              if k not in {"source_year", "source_member", "excel_row", "source_cells"}},
                             sort_keys=True, ensure_ascii=False, default=serializable)
        if ident in seen:
            old_payload, old_row = seen[ident]
            if payload != old_payload:
                raise ValueError("Conflicting official auction identity; do not aggregate")
            duplicate_locations.append({"identity": ident,
                                        "kept": [old_row["source_year"], old_row["excel_row"]],
                                        "duplicate": [row["source_year"], row["excel_row"]]})
        else:
            seen[ident] = (payload, row)
            output.append(row)
    return output, duplicate_locations


def daily_aggregate(components):
    by_date = defaultdict(list)
    for component in components:
        by_date[component["date"]].append(component)
    daily = []
    with localcontext() as context:
        context.prec = 50
        for date, group in sorted(by_date.items()):
            group = sorted(group, key=lambda c: (c["time"], c["auction_name"], c["component_id"]))
            if any(c["contract"] != INCLUDED_CONTRACT for c in group):
                raise ValueError("Contract mixing forbidden")
            volume = sum(c["auction_volume"] for c in group)
            price = sum(c["price"] * c["auction_volume"] for c in group) / volume
            bids = (sum(c["total_bid_volume"] for c in group)
                    if all(c["total_bid_volume"] is not None for c in group) else None)
            cover = bids / volume if bids is not None else None
            def weighted(field):
                return (sum(c[field] * c["auction_volume"] for c in group) / volume
                        if all(c[field] is not None for c in group) else None)
            bidders, successes = weighted("bidder_count"), weighted("successful_bidder_count")
            values = {"price": price, "auction_volume": volume, "total_bid_volume": bids,
                      "cover_ratio": cover, "bidder_count": bidders, "successful_bidder_count": successes}
            daily.append({"date": date, **values, "auction_count": len(group), "contract": INCLUDED_CONTRACT,
                          "auction_names": "|".join(sorted(set(c["auction_name"] for c in group))),
                          "completion_evidence": "|".join(sorted(set(c["completion_evidence"] for c in group))),
                          "missing_fields": "|".join(k for k, v in values.items() if v is None),
                          "component_ids": "|".join(c["component_id"] for c in group)})
    return daily


def partition(date):
    if "2012-01-01" <= date < DEV_END:
        return "development"
    if DEV_END <= date < CONFIRM_END:
        return f"confirmation_{date[:4]}"
    raise ValueError("Unexpected partition date; 2026 is excluded")


def write_new_json(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_new_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: decimal_text(row[k]) if isinstance(row[k], Decimal) or row[k] is None else row[k] for k in fields})


def verified_blob(path, sha):
    blob = path.read_bytes()
    if digest(blob) != sha:
        raise ValueError(f"Frozen identity mismatch: {path.name}")
    return blob


def prepare(archive, scope_path, addendum_path, protocol_path, output):
    scope_blob = verified_blob(scope_path, SCOPE_SHA256)
    addendum_blob = verified_blob(addendum_path, ADDENDUM_SHA256)
    protocol_blob = protocol_path.read_bytes()
    protocol = json.loads(protocol_blob)
    here = Path(__file__).resolve().parent
    for name in ("prepare_eex_data.py", "inspect_eex_source.py"):
        verified_blob(here / name, protocol["source_code_sha256"][name])
    if protocol["scope_sha256"] != SCOPE_SHA256 or protocol["addendum_sha256"] != ADDENDUM_SHA256:
        raise ValueError("Preparation protocol parent identities mismatch")
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite or restart private preparation directory")
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    for name, blob in (("EEX_SOURCE_SCOPE.json", scope_blob), ("EEX_SOURCE_ADDENDUM.json", addendum_blob),
                       ("EEX_PREPARATION_PROTOCOL.json", protocol_blob)):
        (output / name).write_bytes(blob)
    started = {"created_utc": utc(), "protocol_sha256": digest(protocol_blob),
               "scope_sha256": SCOPE_SHA256, "addendum_sha256": ADDENDUM_SHA256,
               "archive_sha256": ARCHIVE_SHA256, "market_fit": False,
               "confirmation_numeric_display": False, "raw_or_derived_quote_publication": False}
    write_new_json(output / "PREPARATION_STARTED.json", started)
    rows, annual_sources = [], []
    for year, member, raw, sheets in source_books(archive):
        hi, header, indices, annual_rows, undated = source_rows(year, member, sheets)
        if undated:
            raise ValueError("Nonempty undated result rows require review")
        has_status = indices["status"] is not None
        if has_status != (year >= 2020):
            raise ValueError("Status schema differs from frozen addendum")
        for row in annual_rows:
            row["_has_status"] = has_status
        rows.extend(annual_rows)
        annual_sources.append({"year": year, "member": member, "bytes": len(raw), "sha256": digest(raw),
                               "header_excel_row": hi + 1, "header": [str(v) if v is not None else "" for v in header],
                               "dated_rows": len(annual_rows)})
    # Complete decoded primary rows are private; original ZIP remains the exact byte source.
    with (output / "all_original_primary_rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=serializable, allow_nan=False) + "\n")
    unique, duplicates = deduplicate(rows)
    components, exclusions = [], []
    for row in unique:
        evidence, reason = validate_completed(row, row["_has_status"])
        if reason:
            exclusions.append({"date": row["date"], "year": row["source_year"], "contract": str(row["contract"]).strip(),
                               "excel_row": row["excel_row"], "source_member": row["source_member"], "reason": reason})
        else:
            components.append(make_component(row, evidence))
    daily = daily_aggregate(components)
    outputs = []
    for group in ("development", "confirmation_2024", "confirmation_2025"):
        folder = output / ("development" if group == "development" else "sealed_confirmation")
        folder.mkdir(exist_ok=True)
        group_daily = [r for r in daily if partition(r["date"]) == group]
        group_components = sorted((c for c in components if partition(c["date"]) == group),
                                  key=lambda c: (c["date"], c["time"], c["auction_name"]))
        daily_path = folder / f"EEX_EUA_T3PA_daily_{group}.csv"
        component_path = folder / f"EEX_EUA_T3PA_components_{group}.csv"
        write_new_csv(daily_path, group_daily, DAILY_FIELDS)
        write_new_csv(component_path, group_components, COMPONENT_FIELDS)
        outputs.append({"partition": group, "daily_path": daily_path.relative_to(output).as_posix(),
                        "daily_sha256": digest(daily_path.read_bytes()), "daily_bytes": daily_path.stat().st_size,
                        "daily_rows": len(group_daily), "min_date": group_daily[0]["date"], "max_date": group_daily[-1]["date"],
                        "component_path": component_path.relative_to(output).as_posix(),
                        "component_sha256": digest(component_path.read_bytes()), "component_rows": len(group_components),
                        "multi_auction_dates": sum(r["auction_count"] > 1 for r in group_daily),
                        "rows_with_missing_fields": sum(bool(r["missing_fields"]) for r in group_daily)})
    audit = {**started, "finished_utc": utc(), "passed": True,
             "original_primary_rows": len(rows), "deduplicated_official_rows": len(unique),
             "duplicate_identity_count": len(duplicates), "duplicate_identity_audit": duplicates,
             "included_component_rows": len(components), "daily_rows": len(daily),
             "annual_included_components": dict(sorted(Counter(c["source_year"] for c in components).items())),
             "annual_daily_rows": dict(sorted(Counter(int(r["date"][:4]) for r in daily).items())),
             "completion_evidence_counts": dict(Counter(c["completion_evidence"] for c in components)),
             "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
             "excluded_contract_counts": dict(Counter(r["contract"] for r in exclusions if r["reason"] == "excluded_contract")),
             "cancelled_dates": [r["date"] for r in exclusions if r["reason"] == "explicit_cancelled"],
             "annual_sources": annual_sources, "outputs": outputs,
             "raw_primary_rows_sha256": digest((output / "all_original_primary_rows.jsonl").read_bytes()),
             "daily_columns": DAILY_FIELDS, "component_columns": COMPONENT_FIELDS,
             "eex_raw_or_derived_quote_redistribution_authorized": False,
             "price_summary_or_model_scores_computed": False,
             "formula_note": "Optional fields propagate any missing component to missing daily field; no zero fill. Bidder counts are auction-volume-weighted averages, not distinct persons across auctions.",
             "phase_note": "T3PA is the report code retained into phase IV; it is not a futures contract. T2PA/EAA2/EAA3 excluded."}
    write_new_json(output / "EXCLUDED_ROW_IDENTITIES.json", exclusions)
    write_new_json(output / "PREPARATION_AUDIT.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--addendum", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = prepare(args.archive, args.scope, args.addendum, args.protocol, args.output)
    print(json.dumps({"passed": audit["passed"], "daily_rows": audit["daily_rows"],
                      "included_component_rows": audit["included_component_rows"],
                      "outputs": audit["outputs"], "exclusion_counts": audit["exclusion_counts"],
                      "market_fit": False, "confirmation_numeric_display": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
