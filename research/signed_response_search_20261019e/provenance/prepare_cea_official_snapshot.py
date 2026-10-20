#!/usr/bin/env python3
"""Rebuild the fixed nationwide CEA daily-total mean snapshot without models.

The Tianjin exchange's public /tshq page directly links this static Body.js.
Only its string literal is decoded; JavaScript is never executed. Raw website
content has no identified redistribution license, so use a fresh directory
outside this checkout. The default output is a temporary local directory.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import datetime as dt
import hashlib
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import re
import tempfile
import urllib.request

PAGE_URL = "https://www.chinatcx.com.cn/tshq"
BODY_URL = "https://img.wanwang.xin/pubsf/10335/10335545/cdn-static-pages/pages/pc/12689_zh-cn.html.Body.js?version=20260930192009"
RAW_SHA = "c3e4c075832f6e5087cf416adecbae9d3c1b61859a6ad09fdf84294197393238"
CSV_SHA = "f9f0e09b7500b2bdb4ec8b16c61d5d5fa2b64de63a24c0cf7bd1d269aa2be75d"
HEADER = ["交易日期", "交易品种", "总成交量（吨）", "总成交额（元）", "成交均价（元/吨）"]


class TableParser(HTMLParser):
    """The small static-table parser is adapted from the audited stage-d helper."""

    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = self.row = self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.table is not None:
                raise ValueError("Nested table requires a new parser audit")
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, help="Existing exact Body.js; otherwise one ordinary public GET")
    parser.add_argument("--out-dir", type=Path, help="Fresh directory outside this checkout; existing directories are refused")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[3]
    if args.out_dir is None:
        out = Path(tempfile.mkdtemp(prefix="cea-official-snapshot-"))
    else:
        out = args.out_dir.expanduser().resolve()
        if out.is_relative_to(checkout):
            raise ValueError("Unlicensed raw and per-date values must stay outside this checkout")
        out.mkdir(parents=True, exist_ok=False)

    if args.raw_file is not None:
        raw = args.raw_file.read_bytes()
        acquisition = {"mode": "offline_exact_raw", "input_path": str(args.raw_file.resolve())}
    else:
        request = urllib.request.Request(BODY_URL, headers={"User-Agent": "public-carbon-research-reproduction"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            acquisition = {"mode": "one_public_get", "url": BODY_URL, "status": response.status}
    actual_raw_sha = hashlib.sha256(raw).hexdigest()
    if actual_raw_sha != RAW_SHA:
        raise ValueError(f"Official snapshot bytes changed: {actual_raw_sha}; do not silently substitute")
    text = raw.decode("utf-8")
    if not text.startswith("document.write("):
        raise ValueError("Unexpected static-body format")
    body = ast.literal_eval(text[len("document.write("):text.rfind(")")])
    if not isinstance(body, str):
        raise ValueError("Expected one string literal")
    tables = TableParser()
    tables.feed(body)
    if len(tables.tables) != 8:
        raise ValueError("Unexpected fixed-snapshot table count")
    table = tables.tables[4]
    if table[0] != HEADER:
        raise ValueError("CEA main-table identity changed")
    rows = table[1:]
    if len(rows) != 1251:
        raise ValueError("Unexpected fixed-snapshot row count")
    mean_rounding_errors = []
    for row in rows:
        if len(row) != 5 or row[1] != "CEA" or not re.fullmatch(r"20\d\d-\d\d-\d\d", row[0]):
            raise ValueError("Unexpected dated CEA row")
        dt.date.fromisoformat(row[0])
        volume, turnover, mean = [float(row[i].replace(",", "")) for i in (2, 3, 4)]
        if not all(math.isfinite(x) for x in (volume, turnover, mean)):
            raise ValueError("Nonfinite total field")
        if volume <= 0 or mean <= 0 or turnover < 0:
            raise ValueError("Invalid total quantity, price or amount")
        mean_rounding_errors.append(abs(mean - turnover / volume))
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Duplicate date; do not silently deduplicate")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["date", "price", "volume_tons", "turnover_yuan"])
    writer.writerows([row[0], row[4].replace(",", ""), row[2].replace(",", ""), row[3].replace(",", "")] for row in rows)
    csv_bytes = stream.getvalue().encode("utf-8")
    actual_csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    if actual_csv_sha != CSV_SHA:
        raise ValueError(f"Canonical bytes do not reproduce: {actual_csv_sha}")
    (out / "tianjin_market_page_body.js").write_bytes(raw)
    target = out / "CEA_OFFICIAL_MEAN.csv"
    target.write_bytes(csv_bytes)
    n = len(rows)
    a = n * 3 // 5
    b = a + n // 5
    dates = [dt.date.fromisoformat(row[0]) for row in rows]
    receipt = {
        "status": "PASS_FIXED_RAW_AND_CANONICAL_BYTES",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "acquisition": acquisition,
        "source_url": PAGE_URL,
        "body_url": BODY_URL,
        "raw_sha256": actual_raw_sha,
        "raw_bytes": len(raw),
        "prepared_path": str(target),
        "prepared_sha256": actual_csv_sha,
        "prepared_bytes": len(csv_bytes),
        "rows": n,
        "unique_dates": n,
        "counts_by_year": dict(sorted(collections.Counter(row[0][:4] for row in rows).items())),
        "date_range": [rows[0][0], rows[-1][0]],
        "A": a,
        "B": b,
        "test_partition_range": [rows[b][0], rows[-1][0]],
        "H20_test_origin_range_inclusive": [b, n - 21],
        "H20_test_origin_dates": [rows[b][0], rows[n - 21][0]],
        "origin_counts_for_L64_H20": {"train": a - 20 - 64, "validation": b - a - 20, "final": b - 20 - 64, "test": n - b - 20},
        "max_calendar_gap_days": max((v - u).days for u, v in zip(dates, dates[1:])),
        "max_abs_published_mean_minus_turnover_over_volume": max(mean_rounding_errors),
        "column_selection": "Table index4, all CEA: daily TOTAL volume/turnover/published mean, source columns2/3/4",
        "target": "Published daily total transaction mean, retained as reported; not closing or listing-only price",
        "changes": "Date sort and comma removal only; every source row retained; no fill or recomputed target",
        "limitations": ["Historical snapshot, not real-time publication vintages", "Complete official trading-calendar coverage not independently certified", "Existing CEA raw/QC exposure is distinct from model scoring"],
        "market_fits": 0,
        "forecast_scores_computed": 0,
        "redistribution_license_identified": False,
    }
    (out / "REPRODUCTION_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
