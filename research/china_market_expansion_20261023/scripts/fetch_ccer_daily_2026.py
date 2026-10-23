#!/usr/bin/env python3
"""Archive official CCER daily notices locally; never redistribute publisher HTML or prices."""
import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import pathlib
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

INDEX_URL = "https://www.ccer.com.cn/wcm/ccer/data/2502lshq.json"
ARTICLE_ROOT = "https://www.ccer.com.cn/wcm/ccer/html/"
HEADERS = {"User-Agent": "Academic-Source-Audit/1.0 (research)"}


def download_notice(row, output):
    url_part = row.get("url", "")
    assert re.fullmatch(r"2502lshq/[0-9]{8}/[0-9]+\.shtml", url_part), url_part
    url = ARTICLE_ROOT + url_part
    rec = {
        "publishedTime": row.get("publishedTime"),
        "title": row.get("title"),
        "url": url,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    time.sleep(0.35)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=20) as response:
            body = response.read(500000)
            if response.read(1):
                raise ValueError("article exceeds 500 kB")
            rec.update(http_status=response.status, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), final_url=response.geturl())
            raw_path = output / "raw" / (url_part.split("/")[-1].replace(".shtml", ".html"))
            raw_path.write_bytes(body)
            rec["local_raw"] = raw_path.name
            decoded = body.decode("utf-8", errors="replace")
            cleaned = re.sub(r"<script\b[^>]*>.*?</script>", " ", decoded, flags=re.S | re.I)
            cleaned = re.sub(r"<style\b[^>]*>.*?</style>", " ", cleaned, flags=re.S | re.I)
            content = " ".join(html.unescape(re.sub(r"<[^>]*>", " ", cleaned)).split())
            # Deliberately do not treat cumulative totals as daily turnover.
            patterns = {
                "volume_text": r"(?:核证自愿减排量成交量|当日成交量)\s*([\d,]+)\s*吨",
                "amount_text": r"(?<!累计)成交额\s*([\d,]+(?:\.\d+)?)\s*元",
                "mean_price_text": r"(?:成交均价)\s*([\d,]+(?:\.\d+)?)\s*元/吨",
            }
            rec["observed"] = {k: (m.group(1) if (m := re.search(p, content)) else None) for k, p in patterns.items()}
            rec["reported_no_trade"] = "无成交" in content
    except urllib.error.HTTPError as exc:
        rec.update(http_status=exc.code, error=str(exc))
    except Exception as exc:
        rec["error"] = repr(exc)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", default="2026")
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--index-file", default=None, help="Optional pre-downloaded official index")
    args = ap.parse_args()
    assert args.year == "2026", "pre-frozen data scope is 2026 only"
    assert 0 < args.max_docs <= 200
    out = pathlib.Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    if args.index_file:
        body = pathlib.Path(args.index_file).read_bytes()
        index_origin = {"source": "pre-downloaded_official_index", "index_url": INDEX_URL}
    else:
        with urllib.request.urlopen(urllib.request.Request(INDEX_URL, headers=HEADERS), timeout=20) as response:
            body = response.read(5000000)
            assert not response.read(1)
        index_origin = {"source": "fetched_official_index", "index_url": INDEX_URL}
    data = json.loads(body)
    candidates = [row for row in data["rows"] if str(row.get("publishedTime", "")).startswith(args.year)]
    if len(candidates) > args.max_docs:
        raise RuntimeError(f"{len(candidates)} eligible docs exceed frozen cap {args.max_docs}")
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(download_notice, row, out) for row in candidates]
        for completed in as_completed(jobs):
            rec = completed.result()
            records.append(rec)
            if len(records) % 20 == 0:
                print(json.dumps({"done": len(records), "total": len(candidates), "errors": sum(bool(x.get("error")) for x in records)}), flush=True)
    records.sort(key=lambda r: (r.get("publishedTime") or "", r.get("url", "")))
    with (out / "local_only_rows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["publishedTime", "url", "http_status", "volume_text", "amount_text", "mean_price_text", "reported_no_trade", "error"])
        w.writeheader()
        for rec in records:
            observed = rec.get("observed", {})
            w.writerow({k: rec.get(k) if k not in observed else observed[k] for k in w.fieldnames})
    audit = {
        "source": "CCER national voluntary reduction market official daily notices",
        "index": dict(index_origin, sha256=hashlib.sha256(body).hexdigest(), bytes=len(body), total_official_index_entries=data.get("total")),
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "year": args.year,
        "eligible_document_count": len(candidates),
        "completed": len(records) == len(candidates),
        "download_success_count": sum(x.get("http_status") == 200 for x in records),
        "download_error_count": sum(x.get("http_status") != 200 for x in records),
        "numeric_daily_volume_count": sum(x.get("observed", {}).get("volume_text") is not None for x in records),
        "numeric_daily_amount_count": sum(x.get("observed", {}).get("amount_text") is not None for x in records),
        "numeric_daily_mean_count": sum(x.get("observed", {}).get("mean_price_text") is not None for x in records),
        "articles": [{k: v for k, v in rec.items() if k != "observed" and k != "local_raw"} for rec in records],
        "license": "Unverified. Raw HTML and transaction rows remain local and are not pushed to public GitHub.",
        "evaluation": {"model_fits": 0, "scored_origins": 0, "confirmation_2026": False},
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "articles"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
