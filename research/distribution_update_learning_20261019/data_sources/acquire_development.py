#!/usr/bin/env python3
"""Acquire the fixed, local-only development panel using only the standard library.

Downloads public Mendeley bytes or accepts an identical local archive, checks
source hashes, then accesses EUA price/volume cells only for 2013 <= date < 2024.
The four legacy development inputs are hash-verified; EU_REPORTED is retained
only in a comparison directory because it duplicates much of the EUA history.
No model fitting, prediction, imputation, price adjustment or calendar filling.

Example (output must be outside the repository and new or empty):
  python -B acquire_development.py --download --output-dir /tmp/carbon-development
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile


ARCHIVE_URL = "https://data.mendeley.com/public-files/datasets/d4bv52fgmd/files/4935504f-cf2d-45c0-add4-d6c20d667a45/file_downloaded"
ARCHIVE_SHA256 = "b4ddba5c3609dfa3c13ed4a0c5998c952225c69dc1bc6c8a8f4c73a5fdfd2862"
ARCHIVE_BYTES = 508904
MEMBER_NAME = "data/期货结算价(..A).xlsx"
MEMBER_SHA256 = "f4d04a394806caa0c4c8f8de02954f41c38dcc3dc3f27a8d28aaa161a6223d5d"
EUA_MAIN_SHA256 = "142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f"
START = dt.date(2013, 1, 1)
END_EXCLUSIVE = dt.date(2024, 1, 1)
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HORIZONS = (5, 20)
YEARS = (2021, 2022, 2023)
TRAIN_START_INDEX = 39  # 40 quote nodes, zero-based origin index.
VALIDATION_ORIGINS = 126
CALIBRATION_ORIGINS = 252
MIN_TRAIN_ORIGINS = 128
LEGACY_COMMIT = "1c67e562e721dd82fff4e4143b97a798a1cf44fa"
LEGACY = {
    "GUANGZHOU": {"filename": "GUANGZHOU_development.csv", "rows": 1434,
                  "sha256": "8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051"},
    "HUBEI": {"filename": "HUBEI_development.csv", "rows": 1408,
              "sha256": "04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797"},
    "SHANGHAI": {"filename": "SHANGHAI_D4_development.csv", "rows": 1507,
                 "sha256": "425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1"},
    "EU_REPORTED": {"filename": "EU_REPORTED_development.csv", "rows": 1515,
                    "sha256": "ca7c192380897076ef57516a11ea2ccefda6ba92ab9823d9aa8f2bb281cefedc"},
}
PRIMARY_MARKETS = ("EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI")
EXCLUSION_REASON = "Predefined before new model fitting: Shanghai has 47 author quotes in 2021 and only 27 L20/H20 same-year mature origins; exclude the entire 2021 Shanghai evaluation, retaining source rows and all audit counts."


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def verify_bytes(blob: bytes, expected: str, label: str) -> None:
    if digest(blob) != expected:
        raise ValueError("Fixed source identity mismatch: " + label)


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_bytes(rows: list[dict[str, str]], fields: list[str]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def finite_number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        value_float = float(value)
    except (ValueError, OverflowError):
        return None
    return value_float if math.isfinite(value_float) else None


def load_development_csv(blob: bytes, expected_rows: int, label: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
    if len(rows) != expected_rows:
        raise ValueError("Unexpected record count: " + label)
    dates = []
    for row in rows:
        date = dt.date.fromisoformat(row["date"])
        if date >= END_EXCLUSIVE:
            raise ValueError("Forbidden non-development date: " + label)
        # Numeric access is strictly below the development date guard.
        price = finite_number(row["price"])
        if price is None or price <= 0:
            raise ValueError("Invalid price in fixed development source; no automatic deletion: " + label)
        dates.append(date)
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError("Duplicate or unordered dates; no automatic repair: " + label)
    return rows


def parse_eua_member(blob: bytes) -> tuple[list[dict[str, str]], dict]:
    verify_bytes(blob, MEMBER_SHA256, MEMBER_NAME)
    with zipfile.ZipFile(io.BytesIO(blob)) as xlsx:
        sheet = ET.fromstring(xlsx.read("xl/worksheets/sheet1.xml"))
    headers, footers, output = [], [], []
    excluded_by_year = collections.Counter()
    retained_source_rows = []
    retained_dates = []
    for row in sheet.findall("s:sheetData/s:row", NS):
        source_row = int(row.attrib["r"])
        if source_row <= 5:
            headers.append({c.attrib["r"]: "".join(t.text or "" for t in c.findall(".//s:t", NS))
                            for c in row.findall("s:c", NS)})
            continue
        # Only date cell A is selected before period eligibility is known.
        date_cell = next((c for c in row.findall("s:c", NS)
                          if c.attrib["r"] == "A" + str(source_row)), None)
        if date_cell is None:
            raise ValueError("Missing date cell in fixed workbook")
        if date_cell.attrib.get("t") == "inlineStr":
            footer = "".join(t.text or "" for t in date_cell.findall(".//s:t", NS))
            if source_row != 5269 or footer != "数据来源：Wind":
                raise ValueError("Unexpected non-date row in fixed workbook")
            footers.append({"source_excel_row": source_row, "text": footer})
            continue
        date_value = date_cell.find("s:v", NS)
        if date_value is None:
            raise ValueError("Missing Excel date value")
        date = (dt.datetime(1899, 12, 30) + dt.timedelta(days=float(date_value.text))).date()
        if not START <= date < END_EXCLUSIVE:
            excluded_by_year[date.year] += 1
            continue
        # No price/volume cell is accessed unless 2013 <= date < 2024.
        cells = {c.attrib["r"]: c for c in row.findall("s:c", NS)}
        values = {}
        for column, name in (("B", "price"), ("C", "volume")):
            cell = cells.get(column + str(source_row))
            element = cell.find("s:v", NS) if cell is not None else None
            values[name] = element.text if element is not None and element.text is not None else ""
        output.append({"date": date.isoformat(), **values})
        retained_dates.append(date)
        retained_source_rows.append(source_row)
    if len(output) != 2839 or len(footers) != 1:
        raise ValueError("Unexpected development partition or footer")
    if retained_dates != sorted(retained_dates) or len(set(retained_dates)) != len(retained_dates):
        raise ValueError("Unexpected date duplication/order; no sorting or repair")
    if sum(n for year, n in excluded_by_year.items() if year < 2013) != 1969:
        raise ValueError("Unexpected pre-2013 date-only count")
    if sum(n for year, n in excluded_by_year.items() if year >= 2024) != 454:
        raise ValueError("Unexpected post-2023 date-only count")
    return output, {"header_rows": headers, "footer_rows": footers,
                    "excluded_dates_by_year": dict(sorted(excluded_by_year.items())),
                    "retained_source_excel_row_first": retained_source_rows[0],
                    "retained_source_excel_row_last": retained_source_rows[-1],
                    "eua_price_volume_access_outside_2013_2023": False,
                    "price_text_preserved": True, "missing_volume_text_preserved": True}


def data_summary(rows: list[dict[str, str]]) -> dict:
    dates = [dt.date.fromisoformat(row["date"]) for row in rows]
    price = [finite_number(row["price"]) for row in rows]
    finite = [p for p in price if p is not None]
    gaps = [(a, b, (b-a).days) for a, b in zip(dates, dates[1:])]
    largest = max(gaps, key=lambda x: x[2])
    result = {"rows": len(rows), "first_date": dates[0].isoformat(), "last_date": dates[-1].isoformat(),
              "annual_rows": dict(sorted(collections.Counter(d.year for d in dates).items())),
              "duplicate_dates": len(dates)-len(set(dates)),
              "weekend_dates": sum(d.weekday() >= 5 for d in dates),
              "largest_calendar_gap": {"from": largest[0].isoformat(), "to": largest[1].isoformat(), "days": largest[2]},
              "price_missing_or_nonfinite": len(rows)-len(finite),
              "price_zero": sum(p == 0 for p in finite), "price_negative": sum(p < 0 for p in finite),
              "consecutive_unchanged_prices": sum(a is not None and b is not None and a == b for a,b in zip(price,price[1:]))}
    if any("volume" in r for r in rows):
        volumes = [finite_number(r.get("volume")) for r in rows]
        result["volume"] = {"missing_or_nonfinite": sum(v is None for v in volumes),
                            "missing_dates": [r["date"] for r, v in zip(rows, volumes) if v is None],
                            "zero": sum(v == 0 for v in volumes),
                            "negative": sum(v is not None and v < 0 for v in volumes),
                            "interpretation": "Preserve reported zero separately from missing. No zero filling; no volume imputation."}
    return result


def period_counts(dates: list[dt.date], lower: dt.date, upper: dt.date, lookback: int) -> dict:
    indices = [i for i, date in enumerate(dates) if lower <= date <= upper]
    result = {"period_start_inclusive": lower.isoformat(), "period_end_inclusive": upper.isoformat(),
              "observed_dates": len(indices), "horizons": {}}
    for horizon in HORIZONS:
        eligible = [i for i in indices if i >= lookback and i+horizon < len(dates)
                    and dates[i+horizon] <= upper]
        result["horizons"][str(horizon)] = {"mature_origins": len(eligible),
            "first_origin_date": dates[eligible[0]].isoformat() if eligible else None,
            "last_origin_date": dates[eligible[-1]].isoformat() if eligible else None,
            "first_label_date": dates[eligible[0]+horizon].isoformat() if eligible else None,
            "last_label_date": dates[eligible[-1]+horizon].isoformat() if eligible else None}
    result["joint_h5_h20_mature_origins"] = result["horizons"]["20"]["mature_origins"]
    return result


def evaluation_counts(dates: list[dt.date], market: str, year: int, lookback: int) -> dict:
    counts = period_counts(dates, dt.date(year,1,1), dt.date(year,12,31), lookback)
    excluded = market == "SHANGHAI" and year == 2021
    counts["excluded_by_predefined_design"] = excluded
    counts["exclusion_reason"] = EXCLUSION_REASON if excluded else None
    counts["used_origins_by_horizon"] = {h: 0 if excluded else value["mature_origins"]
                                           for h, value in counts["horizons"].items()}
    counts["used_joint_h5_h20_origins"] = 0 if excluded else counts["joint_h5_h20_mature_origins"]
    return counts


def block_role(dates: list[dt.date], start: int, end: int, horizon: int) -> dict:
    valid = start >= TRAIN_START_INDEX and end >= start and end+horizon < len(dates)
    result = {"origin_index_start": start, "origin_index_end_inclusive": end,
              "origin_count": max(0, end-start+1), "available": valid}
    if valid:
        result.update({"first_origin_date": dates[start].isoformat(), "last_origin_date": dates[end].isoformat(),
                       "first_history_index": start-TRAIN_START_INDEX,
                       "first_history_date": dates[start-TRAIN_START_INDEX].isoformat(),
                       "first_maturity_index": start+horizon, "last_maturity_index": end+horizon,
                       "first_maturity_date": dates[start+horizon].isoformat(),
                       "last_maturity_date": dates[end+horizon].isoformat(),
                       "target_matured_count": end-start+1,
                       "origin_calendar_span_days_inclusive": (dates[end]-dates[start]).days+1,
                       "origin_to_last_maturity_calendar_span_days_inclusive": (dates[end+horizon]-dates[start]).days+1})
    return result


def formal_block_case(dates: list[dt.date], market: str, year: int, horizon: int) -> dict:
    in_year = [i for i, date in enumerate(dates) if date.year == year]
    if not in_year:
        raise ValueError("Expected evaluation year missing from fixed source")
    test_start = in_year[0]
    calibration_end = test_start-horizon-1
    calibration_start = calibration_end-CALIBRATION_ORIGINS+1
    validation_end = calibration_start-horizon-1
    validation_start = validation_end-VALIDATION_ORIGINS+1
    train_end = validation_start-horizon-1
    roles = {"train": block_role(dates, TRAIN_START_INDEX, train_end, horizon),
             "validation": block_role(dates, validation_start, validation_end, horizon),
             "calibration": block_role(dates, calibration_start, calibration_end, horizon)}
    test_indices = [i for i in in_year if i >= TRAIN_START_INDEX and i+horizon < len(dates)
                    and dates[i+horizon].year == year]
    roles["evaluation"] = block_role(dates, test_indices[0], test_indices[-1], horizon) if test_indices else {
        "origin_count": 0, "target_matured_count": 0, "available": False}
    exclusions = []
    if market == "SHANGHAI" and year == 2021:
        exclusions.append(EXCLUSION_REASON)
    if roles["train"]["origin_count"] < MIN_TRAIN_ORIGINS:
        exclusions.append("Predefined minimum of 128 training origins not met")
    if not all(role["available"] for role in roles.values()):
        exclusions.append("Required history, target maturity or complete role block unavailable")
    boundaries = []
    for previous, following in (("train", "validation"), ("validation", "calibration"), ("calibration", "evaluation")):
        a, b = roles[previous], roles[following]
        if a["available"] and b["available"]:
            passed = a["last_maturity_index"] < b["origin_index_start"]
            boundaries.append({"previous_role": previous, "next_role": following,
                               "last_previous_maturity_index": a["last_maturity_index"],
                               "next_first_origin_index": b["origin_index_start"],
                               "last_previous_maturity_date": a["last_maturity_date"],
                               "next_first_origin_date": b["first_origin_date"], "passed": passed})
            if not passed:
                raise ValueError("A target crosses a frozen role boundary")
    eligible = not exclusions
    return {"market": market, "evaluation_year": year, "horizon_observation_steps": horizon,
            "test_start_index_before_year_end_maturity_filter": test_start,
            "test_start_date": dates[test_start].isoformat(), "quotes_in_evaluation_year": len(in_year),
            "eligible_for_model_evaluation": eligible, "exclusion_reasons": exclusions,
            "minimum_train_origins": MIN_TRAIN_ORIGINS, "roles": roles,
            "used_evaluation_origins": roles["evaluation"]["origin_count"] if eligible else 0,
            "strict_maturity_boundaries": boundaries,
            "calibration_origin_calendar_span_days_inclusive": roles["calibration"].get("origin_calendar_span_days_inclusive"),
            "calibration_through_last_label_calendar_span_days_inclusive": roles["calibration"].get("origin_to_last_maturity_calendar_span_days_inclusive")}


def split_counts(dates_by_market: dict[str, list[dt.date]], lookback: int) -> dict:
    fixed, rolling = {}, {str(year): {} for year in YEARS}
    for market in PRIMARY_MARKETS:
        dates = dates_by_market[market]
        fixed[market] = {
            "train": period_counts(dates, dates[0], dt.date(2018,12,31), lookback),
            "tuning": period_counts(dates, dt.date(2019,1,1), dt.date(2019,12,31), lookback),
            "calibration": period_counts(dates, dt.date(2020,1,1), dt.date(2020,12,31), lookback),
            "evaluation": {str(year): evaluation_counts(dates,market,year,lookback) for year in YEARS}}
        for year in YEARS:
            rolling[str(year)][market] = {
                "train": period_counts(dates, dates[0], dt.date(year-3,12,31), lookback),
                "tuning": period_counts(dates, dt.date(year-2,1,1), dt.date(year-2,12,31), lookback),
                "calibration": period_counts(dates, dt.date(year-1,1,1), dt.date(year-1,12,31), lookback),
                "evaluation": evaluation_counts(dates,market,year,lookback)}
    cases = [formal_block_case(dates_by_market[market], market, year, horizon)
             for market in PRIMARY_MARKETS for year in YEARS for horizon in HORIZONS]
    return {"schema_version": 2, "date_only_counts": True, "price_values_used_for_counts": False,
            "lookback_increments": lookback, "history_price_nodes": lookback+1, "horizons_observation_steps": list(HORIZONS),
            "count_rule": "Origin date belongs to its role/year; its h-step target must mature within the same role/year. History may precede the role but never the dataset start. No calendar filling; no price-dependent eligibility. Overlapping origins are not independent samples.",
            "default_design": "observation_block_expanding_train_validation126_calibration252_by_horizon",
            "formal_observation_blocks": {
                "index_base": 0, "train_start_index": TRAIN_START_INDEX, "history_quote_nodes": TRAIN_START_INDEX+1,
                "validation_origins": VALIDATION_ORIGINS, "calibration_origins": CALIBRATION_ORIGINS,
                "minimum_train_origins": MIN_TRAIN_ORIGINS,
                "rule": "test_start = first quote index of evaluation year; cal_end = test_start-h-1; cal_start = cal_end-251; val_end = cal_start-h-1; val_start = val_end-125; train_end = val_start-h-1; train_start=39. End indices are inclusive. Every earlier role's last target matures strictly before the next role's first origin. Test origin and endpoint belong to the same evaluation year.",
                "scheduled_cases_before_exclusions": len(cases),
                "eligible_cases": sum(case["eligible_for_model_evaluation"] for case in cases),
                "cases": cases},
            "calendar_screening_only": {"fixed_train_2018_tuning_2019_calibration_2020": fixed,
                "expanding_rolling": rolling,
                "rolling_rule": "For evaluation year Y: train through Y-3, tuning Y-2, calibration Y-1. Diagnostic capacity only, not the formal model protocol."},
            "predefined_exclusion": {"market": "SHANGHAI", "evaluation_year": 2021, "reason": EXCLUSION_REASON},
            "rolling_sparse_role_warning": "Calendar-year screening would give Shanghai only 47 quotes / 27 H20 mature origins in 2021 as 2022 calibration or 2023 tuning. Formal observation blocks use 252 calibration and 126 validation origins and may span multiple calendar years; no missing dates/prices are filled."}


def overlap_summary(eua: list[dict[str,str]], eu_reported: list[dict[str,str]]) -> dict:
    a = {row["date"]: row["price"] for row in eua}
    b = {row["date"]: row["price"] for row in eu_reported}
    common = sorted(a.keys() & b.keys())
    # Source identity diagnostic only; neither series is treated as a forecast.
    import decimal
    equal = lambda date: decimal.Decimal(a[date]) == decimal.Decimal(b[date])
    diff = [date for date in common if not equal(date)]
    return {"common_dates": len(common), "exact_decimal_price_matches": sum(equal(date) for date in common),
            "pre_2023_common_dates": sum(date < "2023-01-01" for date in common),
            "pre_2023_exact_matches": sum(date < "2023-01-01" and equal(date) for date in common),
            "nonmatching_dates_count": len(diff), "first_nonmatching_date": diff[0] if diff else None,
            "max_absolute_difference": max(float(abs(decimal.Decimal(a[d])-decimal.Decimal(b[d]))) for d in common),
            "panel_decision": "EU_REPORTED excluded from default targets and mutual-transfer sources; retain only as same-market provenance/sensitivity input. No concatenation, averaging, scale adjustment or deduplication within either original file."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true", help="Download the fixed public Mendeley archive using ordinary HTTPS GET")
    mode.add_argument("--archive-local", type=Path, help="Use already acquired archive bytes with the exact same SHA-256")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty local directory outside this repository")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--legacy-development-dir", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    if output == repo or repo in output.parents:
        parser.error("Raw and row-level prices must remain outside the repository")
    if output.exists() and any(output.iterdir()):
        parser.error("output-dir must be new or empty; existing files are not overwritten")
    legacy_dir = (args.legacy_development_dir or repo / "research/cross_market_innovation_20261017/data").resolve()
    legacy_blobs, legacy_rows = {}, {}
    for market, source in LEGACY.items():
        blob = (legacy_dir / source["filename"]).read_bytes()
        verify_bytes(blob, source["sha256"], source["filename"])
        legacy_blobs[market] = blob
        legacy_rows[market] = load_development_csv(blob, source["rows"], market)
    response_url = None
    if args.download:
        request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "CarbonResearchReproduction/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise ValueError("Public archive download did not return HTTP 200")
            response_url = response.url
            blob = response.read(2_000_001)
        if len(blob) > 2_000_000:
            raise ValueError("Unexpected archive response size")
    else:
        blob = args.archive_local.read_bytes()
    verify_bytes(blob, ARCHIVE_SHA256, "d4 archive")
    if len(blob) != ARCHIVE_BYTES:
        raise ValueError("Unexpected fixed archive byte size")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        member = archive.read(MEMBER_NAME)
    eua, eua_receipt = parse_eua_member(member)
    eua_blob = csv_bytes(eua, ["date", "price", "volume"])
    verify_bytes(eua_blob, EUA_MAIN_SHA256, "2013-2023 EUA development output")
    load_development_csv(eua_blob, 2839, "EUA_LONG")
    rows = {"EUA_LONG": eua, **legacy_rows}
    for subdirectory in ("raw", "development", "comparison_only", "audit"):
        (output / subdirectory).mkdir(parents=True, exist_ok=True)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    (output / "raw/d4_source.zip").write_bytes(blob)
    (output / "development/EUA_LONG_development.csv").write_bytes(eua_blob)
    artifacts = {"EUA_LONG": {"relative_path": "development/EUA_LONG_development.csv", "sha256": digest(eua_blob), "bytes": len(eua_blob)}}
    for market in LEGACY:
        parent = "comparison_only" if market == "EU_REPORTED" else "development"
        relative_path = parent + "/" + market + "_development.csv"
        (output / relative_path).write_bytes(legacy_blobs[market])
        artifacts[market] = {"relative_path": relative_path, "sha256": digest(legacy_blobs[market]),
                             "bytes": len(legacy_blobs[market]), "byte_identical_to_legacy_input": True}
    counts = split_counts({market: [dt.date.fromisoformat(r["date"]) for r in rows[market]]
                           for market in PRIMARY_MARKETS}, TRAIN_START_INDEX)
    json_write(output / "audit/SPLIT_DATE_COUNTS.json", counts)
    manifest = {"schema_version": 1, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "script_sha256": digest(Path(__file__).read_bytes()), "market_fits": 0,
                "primary_markets": list(PRIMARY_MARKETS), "comparison_only": ["EU_REPORTED"],
                "confirmation_numeric_reads": 0, "eex_files_opened": 0, "plos_shenzhen_files_opened": 0,
                "eua_main_period": {"start_inclusive": START.isoformat(), "end_exclusive": END_EXCLUSIVE.isoformat()},
                "d4_source": {"doi": "10.17632/d4bv52fgmd.1", "version": 1, "url": ARCHIVE_URL,
                              "download_final_url": response_url, "sha256": ARCHIVE_SHA256, "bytes": ARCHIVE_BYTES,
                              "member": MEMBER_NAME, "member_sha256": MEMBER_SHA256,
                              "source_mode": "public_https_download" if args.download else "fixed_local_archive",
                              "depositor_license": "CC BY 4.0; underlying Wind/Investing rights not independently certified",
                              "price_definition": "Workbook-reported continuous EUA futures settlement; S5100012",
                              "price_unit": "Workbook: EUR per tonne CO2-equivalent",
                              "volume_definition": "Workbook-reported continuous EUA futures volume; S5100010",
                              "volume_unit": "Workbook: tonnes CO2-equivalent",
                              "definition_limitations": ["Wind vs Investing.com vs 根据新闻整理 source statements conflict", "Exchange, exact expiry, roll rule and historical vintage unavailable", "No claim of official per-date contract verification"]},
                "legacy_inputs": LEGACY, "legacy_fixed_source_commit": LEGACY_COMMIT,
                "legacy_source_manifest": "research/cross_market_innovation_20261017/reproduction/MAIN_DATA_SOURCES.json",
                "legacy_figshare_doi": "10.6084/m9.figshare.28682348.v1",
                "legacy_price_definitions": {"GUANGZHOU": "Author reported quote; previous project matched GDEA endPrice",
                                             "HUBEI": "Author reported quote; Latest vs close semantics unresolved",
                                             "SHANGHAI": "Author reported close; sparse dates, unit and per-date official provenance not fully verified",
                                             "EU_REPORTED": "Author reported EU close; contract/venue/currency/roll unverified; substantially duplicate EUA history"},
                "eua_extraction_receipt": eua_receipt, "artifacts": artifacts,
                "development_summaries": {market: data_summary(rr) for market, rr in rows.items()},
                "duplicate_source_audit": overlap_summary(eua, legacy_rows["EU_REPORTED"]),
                "predefined_evaluation_exclusion": counts["predefined_exclusion"],
                "sparse_rolling_warning": counts["rolling_sparse_role_warning"],
                "handling": {"imputation": False, "rounding": False, "price_based_row_deletion": False,
                             "calendar_fill": False, "cross_source_price_merge": False,
                             "original_price_text_preserved": True, "volume_zero_distinct_from_missing": True},
                "public_release_boundary": "Only code, input URLs/identities, aggregated audits and scientific results may be copied to the public study. Do not copy raw archives, workbooks, row-level prices, point predictions or model weights."}
    json_write(output / "audit/ACQUISITION_MANIFEST.json", manifest)
    print(json.dumps({"prepared": True, "output_dir": str(output), "primary_markets": list(PRIMARY_MARKETS),
                      "rows": {market: len(rows[market]) for market in PRIMARY_MARKETS},
                      "eua_main_sha256": digest(eua_blob), "market_fits": 0,
                      "confirmation_numeric_reads": 0,
                      "split_counts": str(output / "audit/SPLIT_DATE_COUNTS.json"),
                      "acquisition_manifest": str(output / "audit/ACQUISITION_MANIFEST.json")}, indent=2))


if __name__ == "__main__":
    main()
