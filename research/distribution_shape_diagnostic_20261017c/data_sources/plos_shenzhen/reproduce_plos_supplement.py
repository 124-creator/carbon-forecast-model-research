#!/usr/bin/env python3
"""Reproduce only the identified PLOS supplement, with a pinned byte identity.

download: normal public retrieval, one request, no retries or alternate sources.
inspect: only headers/date/cell-presence metadata, never prints prices.
extract: explicitly requested date/price export, no technical indicator columns.

The discovery run used download/inspect only. No extracted price CSV was created
or passed to a model. The current supplement belongs to the republished article;
see DOI 10.1371/journal.pone.0329804 for the publication-version correction.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import urllib.request

URL = ("https://journals.plos.org/plosone/article/file?type=supplementary"
       "&id=10.1371/journal.pone.0296105.s001")
SHA256 = "90c49501afa9beb7370c4409084a28729daa4904a7e7e5de310d0f6bd7a3ba10"
BYTES = 13625856
SHEETS = ("shenzhen", "guangdong", "hubei", "shanghai")
EXPECTED = {
    "shenzhen": (1893, "2013-08-05", "2022-03-11"),
    "guangdong": (1706, "2013-12-18", "2022-03-22"),
    "hubei": (1868, "2014-04-28", "2022-03-23"),
    "shanghai": (1198, "2013-11-26", "2022-03-23"),
}


def write_new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def download(path, receipt_path):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists() or Path(receipt_path).exists():
        raise FileExistsError("Refusing to overwrite an existing file/receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"url": URL, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "attempts": 1, "alternate_source_used": False,
               "expected_sha256": SHA256, "expected_bytes": BYTES}
    try:
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(URL, timeout=30) as response, partial.open("xb") as output:
            receipt["status"] = response.status
            receipt["final_url_without_query"] = response.url.split("?", 1)[0]
            receipt["declared_content_length"] = response.headers.get("content-length")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > 50_000_000:
                    raise ValueError("Response exceeds the fixed 50 MB acquisition bound")
                digest.update(block)
                output.write(block)
        receipt.update(reached_eof=True, bytes=size, sha256=digest.hexdigest())
        if size != BYTES or receipt["sha256"] != SHA256:
            raise ValueError("Publisher response changed: fixed byte identity mismatch")
        partial.rename(path)
        receipt["result"] = "PINNED_FILE_VERIFIED"
    except Exception as exc:
        receipt.update(result="FAILED_NO_RETRY", error=type(exc).__name__+": "+str(exc))
        write_new_json(receipt_path, receipt)
        raise
    write_new_json(receipt_path, receipt)
    return receipt


def _valid_date(cell, datemode, xlrd):
    if cell.ctype == xlrd.XL_CELL_DATE:
        return xlrd.xldate_as_datetime(cell.value, datemode).date()
    if isinstance(cell.value, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d"):
            try:
                return dt.datetime.strptime(cell.value.strip(), fmt).date()
            except ValueError:
                pass
    return None


def inspect(path):
    import xlrd

    path = Path(path)
    raw = path.read_bytes()
    if len(raw) != BYTES or hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Input is not the exact verified complete supplement")
    book = xlrd.open_workbook(file_contents=raw, on_demand=True)
    if tuple(book.sheet_names()) != SHEETS:
        raise ValueError("Unexpected workbook sheets")
    metadata = {
        "source_url": URL, "source_sha256": SHA256, "source_bytes": BYTES,
        "article_doi": "10.1371/journal.pone.0296105",
        "supplement_doi": "10.1371/journal.pone.0296105.s001",
        "version_correction_doi": "10.1371/journal.pone.0329804",
        "article_republication_date": "2025-06-05",
        "correction_notice_publication_date": "2025-08-05",
        "publisher_license": "CC BY 4.0", "extract_fields_only": ["date", "price"],
        "source_columns_zero_based": {"date": 0, "close price": 1},
        "no_price_statistics_computed": True, "no_calendar_fill_or_sort": True,
        "no_zero_price_filter": True, "no_technical_indicator_use": True,
        "sheets": [],
    }
    data = {}
    for name in SHEETS:
        sheet = book.sheet_by_name(name)
        if sheet.cell_value(0, 0) != "indicator" or sheet.cell_value(0, 1) != "close price":
            raise ValueError("Date/price header identity changed")
        records = []
        dated = 0
        for row in range(2, sheet.nrows):
            date = _valid_date(sheet.cell(row, 0), book.datemode, xlrd)
            dated += date is not None
            cell = sheet.cell(row, 1)
            if date is not None and cell.ctype == xlrd.XL_CELL_NUMBER and math.isfinite(cell.value):
                records.append((date.isoformat(), float(cell.value)))
        dates = [pair[0] for pair in records]
        if not dates or any(a >= b for a, b in zip(dates, dates[1:])):
            raise ValueError("Dates must already be strictly increasing and unique")
        signature = (len(records), dates[0], dates[-1])
        if signature != EXPECTED[name]:
            raise ValueError("Mechanical date/count signature mismatch")
        metadata["sheets"].append({
            "sheet": name, "worksheet_rows": sheet.nrows, "worksheet_columns": sheet.ncols,
            "date_rows": int(dated), "valid_date_numeric_price_rows": len(records),
            "dated_rows_without_numeric_price": int(dated - len(records)),
            "first_date": dates[0], "last_date": dates[-1], "duplicate_dates": 0,
            "strictly_increasing": True,
            "year_counts": dict(sorted(collections.Counter(x[:4] for x in dates).items())),
        })
        data[name] = records
    book.release_resources()
    return metadata, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download")
    get.add_argument("--out", type=Path, required=True)
    get.add_argument("--receipt", type=Path, required=True)
    for command in ("inspect", "extract"):
        child = commands.add_parser(command)
        child.add_argument("--workbook", type=Path, required=True)
        child.add_argument("--metadata", type=Path, required=True)
        if command == "extract":
            child.add_argument("--sheet", choices=SHEETS, required=True)
            child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        result = download(args.out, args.receipt)
        print(json.dumps({k: result[k] for k in ("result", "bytes", "sha256")}))
        return
    if args.metadata.exists() or (args.command == "extract" and args.out.exists()):
        raise FileExistsError("Refusing to overwrite an output/metadata receipt")
    metadata, rows = inspect(args.workbook)
    if args.command == "extract":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["date", "price"])
            writer.writerows((date, repr(price)) for date, price in rows[args.sheet])
        metadata["explicit_export"] = {
            "sheet": args.sheet, "rows": len(rows[args.sheet]),
            "csv_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            "fields": ["date", "price"], "model_run_performed": False,
        }
    else:
        metadata["explicit_export"] = None
    write_new_json(args.metadata, metadata)
    print(json.dumps({"status": "MECHANICAL_PASS", "sheets": metadata["sheets"],
                      "price_csv_generated": args.command == "extract"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
