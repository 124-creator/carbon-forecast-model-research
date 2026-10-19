#!/usr/bin/env python3
"""Rebuild the fixed official Tianjin mean-price snapshot; never fit a model.

Raw website content has no identified redistribution license. Use a fresh local
output directory outside this checkout. The default is a temporary directory.
The source is the ordinary static Body.js explicitly linked by /tshq, not an
undocumented API. Decode only its string literal; do not evaluate JavaScript.
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

BODY_URL = "https://img.wanwang.xin/pubsf/10335/10335545/cdn-static-pages/pages/pc/12689_zh-cn.html.Body.js?version=20260930192009"
RAW_SHA = "c3e4c075832f6e5087cf416adecbae9d3c1b61859a6ad09fdf84294197393238"
CSV_SHA = "85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a"
HEADER1 = ["交易日期", "交易品种", "总成交量（吨）", "总成交额（元）", "成交均价（元/吨）"]
HEADER2 = ["挂牌竞价", "当日汇总", "挂牌竞价", "当日汇总", "线上均价", "均价"]


class TableParser(HTMLParser):
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
    parser.add_argument("--raw-file", type=Path, help="Existing exact Body.js; otherwise one public GET")
    parser.add_argument("--out-dir", type=Path, help="Must be a fresh directory outside this checkout")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[3]
    if args.out_dir is None:
        out = Path(tempfile.mkdtemp(prefix="tianjin-official-snapshot-"))
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
    (out / "tianjin_body.js").write_bytes(raw)
    text = raw.decode("utf-8")
    if not text.startswith("document.write("):
        raise ValueError("Unexpected static-body format")
    body = ast.literal_eval(text[len("document.write("):text.rfind(")")])
    if not isinstance(body, str):
        raise ValueError("Expected one string literal")
    tables = TableParser()
    tables.feed(body)
    table = tables.tables[0]
    if table[:2] != [HEADER1, HEADER2]:
        raise ValueError("TJEA header identity changed")
    rows = table[2:]
    if len(rows) != 1077:
        raise ValueError("Unexpected fixed-snapshot row count")
    for row in rows:
        if len(row) != 8 or row[1] != "TJEA" or not re.fullmatch(r"20\d\d-\d\d-\d\d", row[0]):
            raise ValueError("Unexpected dated TJEA row")
        dt.date.fromisoformat(row[0])
        total_volume, total_turnover, total_mean = [float(row[i].replace(",", "")) for i in (3, 5, 7)]
        if not all(math.isfinite(x) for x in (total_volume, total_turnover, total_mean)):
            raise ValueError("Nonfinite total field")
        if total_volume <= 0 or total_mean <= 0 or total_turnover < 0:
            raise ValueError("Invalid total quantity, price or amount")
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Duplicate date; do not silently deduplicate")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["date", "price", "volume_tons", "turnover_yuan"])
    writer.writerows([row[0], row[7].replace(",", ""), row[3].replace(",", ""), row[5].replace(",", "")] for row in rows)
    csv_bytes = stream.getvalue().encode("utf-8")
    actual_csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    if actual_csv_sha != CSV_SHA:
        raise ValueError(f"Canonical bytes do not reproduce: {actual_csv_sha}")
    target = out / "TIANJIN_OFFICIAL_MEAN.csv"
    target.write_bytes(csv_bytes)
    n = len(rows)
    a = int(0.6 * n)
    b = a + int(0.2 * n)
    receipt = {
        "status": "PASS_FIXED_RAW_AND_CANONICAL_BYTES",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "acquisition": acquisition,
        "source_url": "https://www.chinatcx.com.cn/tshq",
        "raw_sha256": actual_raw_sha,
        "prepared_path": str(target),
        "prepared_sha256": actual_csv_sha,
        "prepared_bytes": len(csv_bytes),
        "rows": n,
        "counts_by_year": dict(sorted(collections.Counter(row[0][:4] for row in rows).items())),
        "date_range": [rows[0][0], rows[-1][0]],
        "A": a,
        "B": b,
        "test_partition_start": rows[b][0],
        "H20_test_origin_range": [b, n - 21],
        "H20_test_origin_dates": [rows[b][0], rows[n - 21][0]],
        "column_selection": "TJEA main table daily-total volume, turnover and mean, indices 3/5/7",
        "changes": "Date sort only; no deletion, fill, recomputed target or model fit",
        "market_fits": 0,
        "forecast_scores_computed": 0,
        "redistribution_license_identified": False,
    }
    (out / "REPRODUCTION_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
