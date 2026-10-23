#!/usr/bin/env python3
"""Bounded, polite private archival of the Hubei government public carbon table.

Public website form pagination only; no authentication or anti-bot bypass.
DO NOT commit downloaded HTML or row-level data to public GitHub.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from audit_hubei_government_page import audit

URL = "https://www.hbggzyfwpt.cn/jyxx/tpfq/ggxx"
UA = "Academic-Source-Audit/1.0 (noncommercial data integrity study)"


def crawl(start, end, dest, delay, timeout):
    if not (1 <= start <= end <= 177 and end - start + 1 <= 5):
        raise ValueError("Restrict each run to 1–5 official UI pages within 1–177")
    if delay < 1.5:
        raise ValueError("Minimum inter-request wait is 1.5 seconds")
    dest = Path(dest).resolve()
    if dest.exists():
        raise FileExistsError("Refuse overwrite or resume: use a fresh private directory")
    for ancestor in [dest, *dest.parents]:
        if (ancestor / ".git").exists():
            raise ValueError("Do not write third-party source HTML inside any Git repository")
    dest.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for page in range(start, end + 1):
            data = urllib.parse.urlencode({
                "currentPage": str(page),
                "area": "000",
                "scrollValue": "0",
                "productCode": "",
                "publishTimeType": "5",
                "publishTimeStart": "2000-01-01",
                "publishTimeEnd": "2026-10-23",
            }).encode("utf-8")
            req = urllib.request.Request(URL, data=data, headers={
                "User-Agent": UA,
                "Content-Type": "application/x-www-form-urlencoded",
            })
            rec = {
                "page": page,
                "source_url": URL,
                "method": "POST",
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "failure": None,
            }
            response_bytes = None
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    rec["http_status"] = resp.status
                    rec["final_url"] = resp.geturl()
                    response_bytes = resp.read(250001)
                    if len(response_bytes) > 250000:
                        raise ValueError("unexpected response over 250 kB; preserve page as failure")
            except urllib.error.HTTPError as err:
                rec["http_status"] = err.code
                rec["failure"] = "HTTP_ERROR"
                response_bytes = err.read(250001)
            except Exception as err:
                rec["failure"] = type(err).__name__ + ": " + str(err)

            if response_bytes is not None:
                rec["bytes"] = len(response_bytes)
                rec["sha256"] = hashlib.sha256(response_bytes).hexdigest()
                name = "page_%04d.html" % page
                (dest / name).write_bytes(response_bytes)
                rec["private_snapshot"] = name
                if rec.get("http_status") == 200 and len(response_bytes) <= 250000:
                    try:
                        info = audit(response_bytes)
                        rec["row_count"] = info["row_count"]
                        rec["date_min"] = info["date_min"]
                        rec["date_max"] = info["date_max"]
                        rec["product_counts"] = info["product_counts"]
                        rec["trade_mode_counts"] = info["trade_mode_counts"]
                        rec["ratio_flags_count"] = info["ratio_flag_count"]
                        rec["incomplete_numeric_triples"] = info["incomplete_numeric_triples"]
                        if info["row_count"] == 0:
                            rec["failure"] = "NO_TABLE_ROWS"
                    except Exception as err:
                        rec["failure"] = "SCHEMA_OR_PARSE: " + type(err).__name__ + ": " + str(err)
                elif rec["failure"] is None:
                    rec["failure"] = "HTTP_OR_SIZE_NOT_ACCEPTED"
            records.append(rec)
            if rec["failure"] or rec.get("http_status") in (403, 412, 429):
                break  # Explicit denial/failure is a stop condition, not a bypass target.
            if page < end:
                time.sleep(delay)
    finally:
        receipt = {
            "scope": "one private bounded archival batch, not official exchange feed",
            "source_url": URL,
            "requested_pages": [start, end],
            "completed_success_pages": sum(r["failure"] is None for r in records),
            "attempted_pages": len(records),
            "all_success": len(records) == end - start + 1 and all(r["failure"] is None for r in records),
            "raw_rows_deleted": 0,
            "imputed_values": 0,
            "model_fits": 0,
            "scored_origins": 0,
            "confirmation_2026": False,
            "raw_html_publication_rights_verified": False,
            "records": records,
        }
        (dest / "private_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["all_success"] else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-page", type=int, required=True)
    parser.add_argument("--end-page", type=int, required=True)
    parser.add_argument("--private-dir", required=True, help="Fresh private folder OUTSIDE git repository")
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    return crawl(args.start_page, args.end_page, args.private_dir,
                 args.delay_seconds, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
