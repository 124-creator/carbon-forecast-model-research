"""Independent source-to-CSV audit using exact fractions; never fits a model.

Does not import the acquisition, inspection or preparation implementation.
Mechanically reads sealed source values only for equality checks; prints and
records only pass/fail, counts, date boundaries and byte identities.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re
import warnings
import zipfile

EXPECTED_ARCHIVE = "c3ff6ffaaf348cbd17f61d4726fdbd45a7ead089a0006568a490dd63bb080231"


def norm(value):
    return " ".join(str(value or "").lower().split())


def number(value):
    if value is None or value == "":
        return None
    return Fraction(str(value))


def source_results(archive_path):
    import xlrd
    import openpyxl
    blob = archive_path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == EXPECTED_ARCHIVE
    result, all_count, cancelled, excluded = {}, 0, 0, 0
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/"):
                continue
            year = int(re.search(r"report-(20\d\d)-data\.", name).group(1))
            if name.endswith(".xls"):
                book = xlrd.open_workbook(file_contents=archive.read(name))
                sheet = book.sheet_by_name("Primary Market Auction")
                rows = [[xlrd.xldate_as_datetime(c.value, book.datemode) if c.ctype == 3 else c.value
                         for c in sheet.row(i)] for i in range(sheet.nrows)]
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Workbook contains no default style", category=UserWarning)
                    book = openpyxl.load_workbook(io.BytesIO(archive.read(name)), read_only=True, data_only=True)
                    rows = list(book["Primary Market Auction"].values)
                    book.close()
            header_row = next(i for i, r in enumerate(rows[:10]) if "date" in [norm(v) for v in r]
                              and "contract" in [norm(v) for v in r])
            headers = [norm(v) for v in rows[header_row]]
            def col(*names):
                matches = [j for j, v in enumerate(headers) if v in names]
                assert len(matches) == 1
                return matches[0]
            dc, tc, nc, cc = col("date"), col("time", "auction time"), col("auction name"), col("contract")
            indices = {
                "price": col("auction price €/tco2", "auction price eur/tco2"),
                "auction_volume": col("auction volume tco2", "auction volume"),
                "total_bid_volume": col("total amount of bids"),
                "reported_cover_ratio": col("cover ratio"),
                "bidder_count": col("total number of bidders"),
                "successful_bidder_count": col("number of successful bidders", "number of succesful bidders"),
                "revenue": col("total revenue €", "total revenue eur"),
            }
            sc = col("status") if "status" in headers else None
            assert (sc is not None) == (year >= 2020)
            for source_row, row in enumerate(rows[header_row + 1:], start=header_row + 2):
                if row[dc] in (None, ""):
                    assert not any(v not in (None, "") for v in row)
                    continue
                date = row[dc].date().isoformat() if isinstance(row[dc], dt.datetime) else row[dc].isoformat()
                assert date[:4] == str(year)
                all_count += 1
                if row[cc] != "T3PA":
                    assert row[cc] in ("T2PA", "EAA2", "EAA3")
                    excluded += 1
                    continue
                if sc is not None and norm(row[sc]) == "cancelled":
                    cancelled += 1
                    continue
                if sc is not None:
                    assert norm(row[sc]) == "successful"
                assert not any(isinstance(v, str) and re.search(r"cancel|unsuccess|not successful|not held|failed|failure", v, re.I) for v in row)
                numbers = {k: number(row[c]) for k, c in indices.items()}
                assert all(numbers[k] is not None and numbers[k] > 0 for k in ("price", "auction_volume", "revenue"))
                assert abs(numbers["revenue"] - numbers["price"] * numbers["auction_volume"]) <= Fraction(1, 100)
                if "certificate" in headers:
                    assert row[col("certificate")] == "EUA"
                time = row[tc].time().isoformat() if isinstance(row[tc], dt.datetime) else row[tc].isoformat()
                ident = (date, time, str(row[nc]).strip(), "T3PA")
                observed = {**numbers, "completion_evidence": "explicit_successful" if sc is not None else "inferred_completed",
                            "source_year": year, "source_member": name, "source_excel_row": source_row}
                if ident in result:
                    assert observed == result[ident]
                else:
                    result[ident] = observed
    return result, {"all_original_dated_rows": all_count, "excluded_contract_rows": excluded, "cancelled_rows": cancelled}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_equal(actual, expected):
    if expected is None:
        assert actual == ""
    else:
        # 50-digit Decimal export is checked against independent exact arithmetic.
        assert abs(Fraction(actual) - expected) <= max(Fraction(1), abs(expected)) / 10**42


def verify(archive, prepared):
    source, counts = source_results(archive)
    by_date = defaultdict(list)
    for ident, values in source.items():
        by_date[ident[0]].append((ident, values))
    checks, partitions = 0, []
    for name, begin, end in (("development", "2012-01-01", "2024-01-01"),
                             ("confirmation_2024", "2024-01-01", "2025-01-01"),
                             ("confirmation_2025", "2025-01-01", "2026-01-01")):
        folder = prepared / ("development" if name == "development" else "sealed_confirmation")
        component_file = folder / f"EEX_EUA_T3PA_components_{name}.csv"
        daily_file = folder / f"EEX_EUA_T3PA_daily_{name}.csv"
        components, daily = read_csv(component_file), read_csv(daily_file)
        expected_components = {k: v for k, v in source.items() if begin <= k[0] < end}
        assert len(components) == len(expected_components)
        found = set()
        for row in components:
            ident = tuple(row[k] for k in ("date", "time", "auction_name", "contract"))
            assert ident not in found and ident in expected_components
            found.add(ident)
            expected = expected_components[ident]
            for field in ("price", "auction_volume", "total_bid_volume", "reported_cover_ratio", "bidder_count", "successful_bidder_count", "revenue"):
                check_equal(row[field], expected[field])
                checks += 1
            assert row["completion_evidence"] == expected["completion_evidence"]
            assert row["source_member"] == expected["source_member"]
            assert int(row["source_excel_row"]) == expected["source_excel_row"]
            checks += 3
        expected_dates = sorted(d for d in by_date if begin <= d < end)
        assert [r["date"] for r in daily] == expected_dates
        for row in daily:
            group = by_date[row["date"]]
            volume = sum(v["auction_volume"] for _, v in group)
            bids = sum(v["total_bid_volume"] for _, v in group) if all(v["total_bid_volume"] is not None for _, v in group) else None
            expected = {"price": sum(v["price"] * v["auction_volume"] for _, v in group) / volume,
                        "auction_volume": volume, "total_bid_volume": bids,
                        "cover_ratio": bids / volume if bids is not None else None}
            for field in ("bidder_count", "successful_bidder_count"):
                expected[field] = (sum(v[field] * v["auction_volume"] for _, v in group) / volume
                                   if all(v[field] is not None for _, v in group) else None)
            for field, value in expected.items():
                check_equal(row[field], value)
                checks += 1
            assert row["contract"] == "T3PA"
            assert row["auction_names"] == "|".join(sorted(set(k[2] for k, _ in group)))
            assert int(row["auction_count"]) == len(group)
            assert set(row["missing_fields"].split("|")) - {""} == {k for k, v in expected.items() if v is None}
            checks += 4
        partitions.append({"partition": name, "daily_rows": len(daily), "component_rows": len(components),
                           "min_date": expected_dates[0], "max_date": expected_dates[-1],
                           "daily_sha256": hashlib.sha256(daily_file.read_bytes()).hexdigest(),
                           "components_sha256": hashlib.sha256(component_file.read_bytes()).hexdigest()})
    return {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
            "archive_sha256": EXPECTED_ARCHIVE,
            "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_implementation": "Direct xlrd/openpyxl reads and exact fractions. No import of preparation/inspection/acquisition code.",
            "numeric_and_metadata_comparisons": checks, "counts": counts, "partitions": partitions,
            "only_mechanical_source_fidelity_checks": True, "confirmation_numeric_values_displayed": False,
            "confirmation_price_distributions_computed": False, "market_fit": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Independent audit receipt exists")
    receipt = verify(args.archive, args.prepared)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
