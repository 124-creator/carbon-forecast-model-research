"""Reproduce the frozen numeric parser, then apply one source-certified footnote.

No realized event table is read here. Only the SHA-pinned 2019-12-12 source's
T3PA / CAP auction rows explicitly carrying *** receive the announced deduction.
All vintage dates, missing-version gaps, identities and other quantities persist.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


SOURCE_SHA = "22820d037bac47c38264732b87b0ec39f648259df2051d2bdbf63f6008daf383"
ORIGINAL_PARSER_SHA = "e336bbf77580109851895e3812df8f537ae13b4749d8febfce1b5fa5a1ee40c2"
ORIGINAL_CALENDAR_SHA = "710017cf3f3096385c7c909659a2a376a067ac2200b7081825e4c55d24e38411"
RULE = "EEX_2020_v2.1_explicit_triple_star_IF_postponement"
DEDUCTION = 373000
EXPECTED_NOTE = "***) The volumes for the Innovation Fund (373,000 allowances in each EU Auction on Mondays, Tuesdays and Thursdays) shall be postponed"
EXPECTED_CONTINUATION = "until necessary practical arrangements are in place. See Customer Infomation dated 12 December 2019."
EXTRA_FIELDS = ["plannedvolume_numeric_raw", "plannedvolume_text_adjustment",
                "plannedvolume_rule_id", "plannedvolume_evidence", "volume_interpretation"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames), rows


def source_events(module, book):
    """Calendar numeric cells and markers, independent of any realized quantity."""
    out = []
    columns = None
    for index, row in enumerate(book["rows"]):
        normal = [str(v).strip().lower() if v is not None else "" for v in row]
        if "product code" in normal and "volume" in normal:
            names = [i for i, value in enumerate(normal) if value in ("auction name", "auctionname")]
            if names:
                columns = (normal.index("product code"), normal.index("volume"), names[0])
            continue
        if not columns or str(row[columns[0]]).strip() != "T3PA":
            continue
        scheduled = module.as_date(row[0])
        if scheduled is None and index + 1 < len(book["rows"]):
            scheduled = module.as_date(book["rows"][index + 1][0])
        if scheduled is None or scheduled.year != book["year"]:
            continue
        name = str(row[columns[2]] or "").strip()
        volume, _ = module.volume_of(row[columns[1]])
        out.append({"scheduleddate": scheduled.isoformat(), "platform": module.platform_of(name),
                    "auction_name": name, "numeric_volume": volume, "source_row": index + 1,
                    "triple_star": bool(re.search(r"(?<!\*)\*{3}(?!\*)", name))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    study = args.study_dir.resolve()
    correction = Path(__file__).resolve().parent
    output = args.output_dir.resolve() if args.output_dir else correction / "rerun/local_only"
    if study == output or study / "local_only" == output:
        raise ValueError("Refuse to overwrite the v1 source directory")
    output.mkdir(parents=True, exist_ok=True)
    evidence_dir = correction / "local_only"
    evidence_dir.mkdir(exist_ok=True)
    original_parser = study / "parse_calendar_vintages.py"
    original_calendar = study / "local_only/calendar_vintages_development.csv"
    assert sha(original_parser) == ORIGINAL_PARSER_SHA
    assert sha(original_calendar) == ORIGINAL_CALENDAR_SHA
    fields, original_rows = read_csv(original_calendar)
    spec = importlib.util.spec_from_file_location("frozen_calendar_parser", original_parser)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Reproduce the original transformation from only authorized calendar years.
    numeric_dir = evidence_dir / "reproduced_numeric_only"
    result = subprocess.run([sys.executable, str(original_parser), "--source-root", str(module.DEFAULT_SOURCE),
                             "--output-dir", str(numeric_dir), "--min-year", "2017", "--max-year", "2023"],
                            text=True, capture_output=True, check=True)
    (evidence_dir / "numeric_parser_stdout.txt").write_text(result.stdout, encoding="utf-8")
    assert sha(numeric_dir / "calendar_vintages_development.csv") == ORIGINAL_CALENDAR_SHA

    source_paths = {r["source_path"] for r in original_rows if r["source_sha256"] == SOURCE_SHA}
    assert len(source_paths) == 1
    source = Path(next(iter(source_paths)))
    assert sha(source) == SOURCE_SHA
    book = module.load_workbook_metadata(source)
    assert book["sheet"] == "Calendar"
    assert module.as_date(book["rows"][13][0]).isoformat() == "2019-12-12"
    assert str(book["rows"][13][1]) == "2.1"
    assert str(book["rows"][33][0]) == EXPECTED_NOTE
    assert str(book["rows"][34][0]).strip() == EXPECTED_CONTINUATION
    events = source_events(module, book)
    eligible = {r["source_row"]: r for r in events if r["platform"] == "CAP" and r["triple_star"]}
    assert len(eligible) == 96
    assert all(r["scheduleddate"] <= "2020-08-31" for r in eligible.values())
    adjusted, changes, untouched_indicative = [], [], []
    for original in original_rows:
        row = dict(original)
        raw_volume = int(row["plannedvolume"])
        row.update(plannedvolume_numeric_raw=raw_volume, plannedvolume_text_adjustment=0,
                   plannedvolume_rule_id="", plannedvolume_evidence="",
                   volume_interpretation="numeric_cell")
        if row["source_sha256"] == SOURCE_SHA:
            source_row = int(row["source_row"])
            assert row["revision_available_date"] == "2019-12-13"
            assert row["valid_until_exclusive"] == "2020-03-18"
            if row["product_code"] == "T3PA" and row["platform"] == "CAP" and source_row in eligible:
                event = eligible[source_row]
                assert row["scheduleddate"] == event["scheduleddate"]
                assert row["auction_name"] == event["auction_name"]
                assert raw_volume == event["numeric_volume"]
                corrected = raw_volume - DEDUCTION
                assert corrected > 0 and corrected % 500 == 0
                row.update(plannedvolume=corrected, plannedvolume_text_adjustment=-DEDUCTION,
                           plannedvolume_rule_id=RULE,
                           plannedvolume_evidence="Calendar!A14:C14; Calendar!A34:A35; explicit *** auction-name marker",
                           volume_interpretation="numeric_cell_minus_announced_postponed_volume")
                changes.append({"vintage_id": row["vintage_id"], "scheduleddate": row["scheduleddate"],
                                "platform": row["platform"], "source_row": source_row,
                                "source_sha256": SOURCE_SHA, "auction_name": row["auction_name"],
                                "original_numeric_volume": raw_volume, "text_adjustment": -DEDUCTION,
                                "corrected_plannedvolume": corrected, "rule_id": RULE})
            elif row["platform"] == "CAP":
                row["volume_interpretation"] = "indicative_unmarked_scope_not_extended"
                untouched_indicative.append(row["scheduleddate"])
        adjusted.append(row)
    assert len(changes) == 96 and len(untouched_indicative) == 44
    destination = output / "calendar_vintages_development.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + EXTRA_FIELDS)
        writer.writeheader()
        writer.writerows(adjusted)
    with (evidence_dir / "CALENDAR_CHANGED_ROWS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)

    # Audit the surrounding 2020 vintages plus correction-like notes in all
    # 34 development XLSX overviews. Never read archive confirmation-year files.
    source_audit = []
    surrounding = []
    for path in sorted(module.DEFAULT_SOURCE.rglob("*.xlsx")):
        match = re.search(r"/History_EUA_Primary_Auction_Calendar_(\d{4})/", str(path), re.I)
        if not match or not 2017 <= int(match[1]) <= 2023:
            continue
        b = module.load_workbook_metadata(path)
        matches = []
        for ri, rr in enumerate(b["rows"][:100], 1):
            for ci, value in enumerate(rr):
                if isinstance(value, str) and any(term in value.lower() for term in
                    ("postpon", "innovation fund", "deduct", "reduced by", "subtract", "redistribution", "indicative")):
                    matches.append({"row": ri, "column_index_1based": ci + 1, "text": value})
        source_audit.append({"source_path": b["path"], "source_sha256": b["sha256"],
                             "year": b["year"], "sheet": b["sheet"], "relevant_cells": matches})
        if b["year"] == 2020:
            es = source_events(module, b)
            cap = [e for e in es if e["platform"] == "CAP"]
            jan9 = [e for e in cap if e["scheduleddate"] == "2020-01-09"]
            accepted = [r for r in original_rows if r["source_sha256"] == b["sha256"]]
            surrounding.append({"source_path": b["path"], "source_sha256": b["sha256"],
                                "document_history": b["history"], "relevant_cells": matches,
                                "cap_count": len(cap), "triple_star_cap_count": sum(e["triple_star"] for e in cap),
                                "numeric_jan9_example": jan9,
                                "accepted_by_frozen_parser": bool(accepted),
                                "revision_available_date": accepted[0]["revision_available_date"] if accepted else None,
                                "valid_until_exclusive": accepted[0]["valid_until_exclusive"] if accepted else None})
    dump(evidence_dir / "SOURCE_FOOTNOTE_AUDIT.json", source_audit)
    dump(evidence_dir / "SURROUNDING_2020_VINTAGES.json", surrounding)
    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "rule_id": RULE,
        "source_path": str(source), "source_sha256": SOURCE_SHA,
        "source_sheet": "Calendar", "revision_cells": "A14:C14", "footnote_cells": "A34:A35",
        "footnote_text": [EXPECTED_NOTE, EXPECTED_CONTINUATION],
        "revision_date": "2019-12-12", "revision_available_date": "2019-12-13",
        "valid_until_exclusive": "2020-03-18", "deduction_per_explicit_row": DEDUCTION,
        "original_parser_sha256": ORIGINAL_PARSER_SHA, "original_csv_sha256": ORIGINAL_CALENDAR_SHA,
        "original_csv_reproduced_byte_exact": True, "corrected_csv_sha256": sha(destination),
        "row_count": len(adjusted), "vintage_count": len({r["vintage_id"] for r in adjusted}),
        "adjusted_calendar_rows": len(changes), "added_provenance_columns": EXTRA_FIELDS,
        "adjusted_schedule_date_range": [min(r["scheduleddate"] for r in changes), max(r["scheduleddate"] for r in changes)],
        "unmarked_indicative_CAP_rows_not_adjusted": len(untouched_indicative),
        "unmarked_scope_limitation": "The 44 September-December CAP rows carry only the MSR * marker; this narrow correction does not extrapolate *** to them. Current frozen horizons while this vintage is visible do not select them.",
        "scope": "2017-2023 overview calendar XLSX only; no 2024/2025 calendar contents, no realized auction outcomes read by this parser",
        "source_workbooks_audited": len(source_audit),
        "ambiguity_preserved": "The March17 and April3 2020 files share an ambiguous March17 Document History date and remain excluded; the [2020-03-18,2020-07-02) availability gap persists.",
        "later_version_use": "July1/September 2020 versions are separate later information; their numerical revisions and July16 Innovation Fund start do not set or backdate this rule.",
        "all_nonvolume_original_fields_unchanged": all(all(str(a[k]) == str(b[k]) for k in fields if k != "plannedvolume") for a, b in zip(original_rows, adjusted)),
        "outcome_based_rule": False, "training_or_scoring": False,
    }
    assert receipt["all_nonvolume_original_fields_unchanged"]
    dump(correction / "CALENDAR_CORRECTION_RECEIPT.json", receipt)
    coverage = json.loads((numeric_dir / "CALENDAR_VINTAGE_COVERAGE.json").read_text())
    coverage.update(output_sha256=sha(destination), volume_correction_rule=RULE,
                    correction_receipt=str(correction / "CALENDAR_CORRECTION_RECEIPT.json"))
    dump(output / "CALENDAR_VINTAGE_COVERAGE.json", coverage)
    print(json.dumps({k: receipt[k] for k in ("adjusted_calendar_rows", "row_count", "vintage_count", "corrected_csv_sha256", "all_nonvolume_original_fields_unchanged")}))


if __name__ == "__main__":
    main()
