"""Parse saved official reports, preserving missingness and literal source values.

No corrections are made to OHLC containment, no zero fill is performed, and
reports for 2024 onward are rejected. A source-explicit absence of listed trades
can set listed_volume=0; an absent numeric field remains empty/None.
"""
from pathlib import Path
import argparse
from prepare_eua_sources import require_non_git_path
import csv
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import html
import json
import re

ROOT = Path("/tmp/carbon-cea-official-pre2024")


class BodyParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div" and "article-con" in attrs.get("class", "").split():
            self.depth = 1
            return
        if self.depth:
            if tag == "div":
                self.depth += 1
            if tag in ("p", "br", "tr"):
                self.parts.append("\n")
            if tag == "img":
                self.images.append(attrs.get("src"))

    def handle_endtag(self, tag):
        if self.depth:
            if tag == "div":
                self.depth -= 1
            if tag in ("p", "br", "tr"):
                self.parts.append("\n")

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def number(text):
    return float(text.replace(",", "").replace("，", ""))


def numeric(pattern, body):
    found = re.search(pattern, body)
    if not found:
        return None, None
    value = number(found.group(1))
    if len(found.groups()) >= 2 and found.group(2) == "万":
        value *= 10_000
    return value, found.group(0)


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = require_non_git_path(args.output_dir)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    index = json.loads((ROOT / "report_index.json").read_text(encoding="utf-8"))
    rows, evidence = [], []
    for meta in index:
        date = meta["date"]
        assert "2021-01-01" <= date < "2024-01-01"
        path = ROOT / "reports" / f"{date}_{meta['article_id']}.html"
        if not path.exists():
            continue
        raw = path.read_bytes()
        parser = BodyParser()
        parser.feed(raw.decode("utf-8"))
        body = re.sub(r"[\t\r ]+", " ", "".join(parser.parts))
        body = re.sub(r"\n\s*\n", "\n", body).strip()
        title_date = re.search(r"(20\d{6})\s*$", meta["title"])
        row = {
            "date": date,
            "source_title_date": (
                datetime.strptime(title_date.group(1), "%Y%m%d").strftime("%Y-%m-%d")
                if title_date else None
            ),
            "price_scope": "composite_price" if "综合价格行情" in body else "CEA_listed_price",
        }
        snippets = {}
        for field, label in [("open", "开盘价"), ("high", "最高价"), ("low", "最低价"), ("close", "收盘价")]:
            value, snippet = numeric(label + r"[：:\s]*([0-9,，]+(?:\.[0-9]+)?)\s*元/吨", body)
            row[field] = value
            snippets[field] = snippet
        volume, snippet = numeric(
            r"挂牌协议交易[^。；;]*?成交量\s*([0-9,，]+(?:\.[0-9]+)?)\s*(万)?吨", body
        )
        if volume is None and re.search(r"(?:今日)?无挂牌协议交易", body):
            volume, snippet = 0.0, "无挂牌协议交易"
        row["listed_volume_tonnes"] = volume
        snippets["listed_volume_tonnes"] = snippet
        amount, snippet = numeric(
            r"挂牌协议交易[^。；;]*?成交额\s*([0-9,，]+(?:\.[0-9]+)?)\s*(万)?元", body
        )
        row["listed_amount_cny"] = amount
        snippets["listed_amount_cny"] = snippet
        row["source_url"] = meta["url"]
        row["source_html_sha256"] = hashlib.sha256(raw).hexdigest()
        row["source_file"] = str(path.relative_to(ROOT))
        rows.append(row)
        evidence.append({**meta, "values": row, "matched_fragments": snippets,
                         "article_text": body, "images": parser.images})

    rows.sort(key=lambda row: (row["date"], row["source_url"]))
    destination = ROOT / "cea_official_pre2024.csv"
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "parsed_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ("open", "high", "low", "close", "listed_volume_tonnes")
    complete = [r for r in rows if all(r[k] is not None for k in ("open", "high", "low", "close"))]
    violation = [r for r in complete if not r["low"] <= min(r["open"], r["close"]) <= max(r["open"], r["close"]) <= r["high"]]
    zero = [r for r in rows if r["listed_volume_tonnes"] == 0]
    audit = {
        "parser_version": "1", "row_count": len(rows),
        "date_min": rows[0]["date"], "date_max": rows[-1]["date"],
        "years": {str(y): sum(r["date"].startswith(str(y)) for r in rows) for y in (2021, 2022, 2023)},
        "date_unique_count": len({r["date"] for r in rows}),
        "missing_counts": {k: sum(r[k] is None for r in rows) for k in fields},
        "nonpositive_ohlc_dates": [r["date"] for r in complete if min(r[k] for k in ("open", "high", "low", "close")) <= 0],
        "source_title_date_disagreements": [r["date"] for r in rows if r["source_title_date"] != r["date"]],
        "ohlc_containment_violation_dates": [r["date"] for r in violation],
        "zero_listed_volume_dates": [r["date"] for r in zero],
        "scope_first_dates": {scope: min(r["date"] for r in rows if r["price_scope"] == scope) for scope in {r["price_scope"] for r in rows}},
        "csv_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "warnings": ["Not claimed complete until all archive pages are fetched.",
                     "Missing numeric fields are None/empty, never fabricated zero.",
                     "Composite prices are a distinct reporting regime; no OHLC repair is performed."],
    }
    (ROOT / "parse_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
