#!/usr/bin/env python3
"""Verify an authorized 2018–2025 SCEEX archive; optionally rebuild privately.

Default behavior is READ ONLY. Never upload generated source rows to a public repo.
This script checks source-page hashes and EVERY data-row count, not only artifact SHA.
"""
import argparse
import collections
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

HEAD = ["交易日期", "交易机构", "交易品种", "开盘价", "最高价",
        "最低价", "成交均价", "收盘价", "成交量", "成交金额"]
FIELDS = ["date", "institution", "instrument", "open_text", "high_text",
          "low_text", "mean_text", "close_text", "volume_text", "turnover_text"]
OUTPUT_FIELDS = FIELDS + ["source_file", "source_table_row", "source_url"]


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.tables = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.tables.append([])
        elif self.depth == 1 and tag == "tr":
            self.row = []
        elif self.depth == 1 and tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if self.depth == 1 and tag in ("td", "th") and self.cell is not None:
            if self.row is not None:
                self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif self.depth == 1 and tag == "tr" and self.row is not None:
            self.tables[-1].append(self.row)
            self.row = None
        elif tag == "table":
            self.depth -= 1


def checked_source_rows(root):
    metadata = json.loads((root / "COVERAGE_METADATA.json").read_text(encoding="utf-8"))
    recovered = []
    per_page = []
    for p in metadata["pages"]:
        idx = int(p["page"])
        name = f"sceex_domestic_2018_2025_page{idx:04d}.html"
        raw = (root / "raw" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == p["sha256"], f"raw page {idx} SHA mismatch"
        receipt = json.loads((root / "receipts" / (name + ".json")).read_text(encoding="utf-8"))
        assert receipt.get("http_status") == 200
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        tables = Tables()
        tables.feed(raw.decode("utf-8"))
        candidates = [table for table in tables.tables if table and table[0] == HEAD]
        assert len(candidates) == 1, f"ambiguous source table: {idx}"
        data = candidates[0][1:]
        assert len(data) == int(p["rows"]), f"source page row-count mismatch: {idx}"
        for j, values in enumerate(data, 1):
            assert len(values) == len(FIELDS)
            r = dict(zip(FIELDS, values))
            r.update(source_file=name, source_table_row=str(j), source_url=receipt["requested_url"])
            assert "2018-01-01" <= r["date"] <= "2025-12-31"
            recovered.append(r)
        per_page.append({"page": idx, "reparsed": len(data)})
    assert len(recovered) == metadata["all_rows"] == sum(x["reparsed"] for x in per_page)
    return recovered, per_page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-root", type=Path, required=True)
    ap.add_argument("--private-out", type=Path, default=None,
                    help="Optional path OUTSIDE public git repo; refuses to overwrite.")
    args = ap.parse_args()
    root = args.archive_root.resolve()
    rows, per_page = checked_source_rows(root)
    old = root / "parsed" / "sceex_domestic_2018_2025_all_reported.csv"
    with old.open(encoding="utf-8", newline="") as stream:
        stored = list(csv.DictReader(stream))
    is_prefix = len(stored) <= len(rows) and all(
        all(a.get(k) == b[k] for k in OUTPUT_FIELDS)
        for a, b in zip(stored, rows)
    )
    verdict = {
        "status": "PASS" if len(stored) == len(rows) and is_prefix else "FAIL_INCOMPLETE_CANONICAL",
        "source_pages": len(per_page),
        "raw_rows": len(rows),
        "stored_csv_rows": len(stored),
        "missing_from_canonical": len(rows) - len(stored),
        "stored_is_true_prefix": is_prefix,
        "per_page_source_counts": per_page,
        "models_fitted": 0,
        "forecast_scores": 0,
    }
    if args.private_out is not None:
        dest = args.private_out.resolve()
        if dest.exists():
            raise FileExistsError(f"Preserve old artifacts: refusing overwrite {dest}")
        if dest.is_relative_to(root):
            raise ValueError("Private recovered output must be outside the source archive")
        for parent in [dest.parent, *dest.parent.parents]:
            if (parent / ".git").exists():
                raise ValueError("Refusing to write market rows inside a git working tree")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with dest.open(encoding="utf-8", newline="") as stream:
            assert sum(1 for _ in csv.DictReader(stream)) == len(rows)
        verdict["private_recovery"] = {
            "path": str(dest), "bytes": dest.stat().st_size,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "rows": len(rows),
        }
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict["status"] == "PASS" or args.private_out else 2


if __name__ == "__main__":
    raise SystemExit(main())
