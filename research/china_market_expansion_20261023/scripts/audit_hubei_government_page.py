#!/usr/bin/env python3
"""Read-only audit of the Hubei provincial carbon trading disclosure table.
NO forecasts; NO inferred price units; preserves malformed/unavailable cells.
Usage: python audit_hubei_government_page.py /private/hubei_snapshot.html
"""
import argparse
import collections
from decimal import Decimal, InvalidOperation
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

HEADERS = ["成交日期", "交易产品", "交易价格", "交易方式", "成交总量", "成交总价", "日收益价"]
COLUMNS = ["trade_date", "product", "trade_price_text", "mode",
           "volume_text", "amount_text", "daily_yield_price_text"]

class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.level = 0
        self.tables = []
        self.row = None
        self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.level += 1
            if self.level == 1:
                self.tables.append([])
        if self.level == 1 and tag == "tr":
            self.row = []
        if self.level == 1 and tag in ("td", "th"):
            self.cell = []
    def handle_data(self, value):
        if self.cell is not None:
            self.cell.append(value)
    def handle_endtag(self, tag):
        if self.level == 1 and tag in ("td", "th") and self.cell is not None:
            if self.row is not None:
                self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if self.level == 1 and tag == "tr" and self.row is not None:
            self.tables[-1].append(self.row)
            self.row = None
        if tag == "table":
            self.level -= 1

def dec(value):
    if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", value.replace(",", "")):
        return None
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None

def audit(raw):
    html = Tables()
    html.feed(raw.decode("utf-8", errors="replace"))
    candidates = [t for t in html.tables if t and t[0] == HEADERS]
    if len(candidates) != 1:
        raise ValueError("ambiguous or missing exact seven-column Hubei table")
    raw_rows = candidates[0][1:]
    if any(len(r) != len(COLUMNS) for r in raw_rows):
        raise ValueError("source table row width changed; do not drop rows")
    vals = [dict(zip(COLUMNS, r)) for r in raw_rows]
    counts = collections.Counter()
    duplicate_keys = collections.Counter((x["trade_date"], x["product"], x["mode"]) for x in vals)
    flags = []
    for x in vals:
        if not re.fullmatch(r"\d{8}", x["trade_date"]):
            counts["invalid_date"] += 1
        volume = dec(x["volume_text"])
        amount = dec(x["amount_text"])
        price = dec(x["daily_yield_price_text"])
        if x["trade_price_text"] == "600000":
            counts["uninterpreted_trade_price_literal_600000"] += 1
        if volume is None or amount is None or price is None:
            counts["incomplete_numeric_triples"] += 1
        elif volume <= 0:
            counts["nonpositive_volume"] += 1
        else:
            counts["positive_finite_triples"] += 1
            ratio = amount / volume
            if abs(price - ratio) > max(Decimal("0.05"), Decimal("0.001") * abs(ratio)):
                flags.append({"trade_date": x["trade_date"], "product": x["product"],
                              "source_table_row_index": vals.index(x) + 1,
                              "type": "daily_yield_price_vs_amount_over_volume"})
    return {
        "source": "https://www.hbggzyfwpt.cn/jyxx/tpfq/ggxx",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "row_count": len(vals),
        "date_min": min((x["trade_date"] for x in vals), default=None),
        "date_max": max((x["trade_date"] for x in vals), default=None),
        "product_counts": dict(collections.Counter(x["product"] for x in vals)),
        "trade_mode_counts": dict(collections.Counter(x["mode"] for x in vals)),
        "literal_trade_price_600000_count": counts["uninterpreted_trade_price_literal_600000"],
        "positive_finite_triples": counts["positive_finite_triples"],
        "incomplete_numeric_triples": counts["incomplete_numeric_triples"],
        "nonpositive_volume": counts["nonpositive_volume"],
        "invalid_date": counts["invalid_date"],
        "duplicate_date_product_mode_excess": sum(n - 1 for n in duplicate_keys.values()),
        "ratio_flag_count": len(flags),
        "ratio_flags": flags,
        "interpretation": "The column 交易价格 is NOT certified as a yuan/ton price; 600000 must not be used as the daily carbon price. Numerical ratio checks do not certify channel, units, or point-in-time status.",
        "removed_rows": 0,
        "filled_missing_values": 0,
        "model_fits": 0,
        "scored_origins": 0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("private_html", help="Download an authorized raw snapshot privately first.")
    args = parser.parse_args()
    result = audit(Path(args.private_html).read_bytes())
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
