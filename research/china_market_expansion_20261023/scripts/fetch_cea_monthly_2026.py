#!/usr/bin/env python3
"""Download official CNEEEX CEA 2026 monthly notices into SHA/aggregate receipts.

Do not redistribute the exchange page HTML or treat monthly totals as daily bars.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request

INDEX = "https://overview.cneeex.com/qgtpfqjy/jysj/ydgk/2026n/"
USER_AGENT = "CarbonVolatilityResearch/1.0 academic source audit"


def retrieve(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=25) as response:
        data = response.read(500000)
        return {
            "url": response.url,
            "status": response.status,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "_raw": data,
        }


def month_record(path):
    url = urllib.parse.urljoin(INDEX, path)
    try:
        r = retrieve(url)
        content = r.pop("_raw").decode("utf-8", "replace")
        without_tags = re.sub(r"<[^>]*>", " ", content)
        condensed = re.sub(r"\s+", "", html.unescape(without_tags))
        m = re.search(r"本月(?:全国碳排放配额|全国碳排放权交易市场配额)总成交量([\d,]+)吨[，,]总成交额([\d,]+(?:\.\d+)?)元", condensed)
        r["publication_month"] = re.search(r"/c/(\d{4}-\d{2})-\d{2}/", path).group(1)
        r["volume_tonnes_month"] = int(m.group(1).replace(",", "")) if m else None
        r["turnover_cny_month_as_text"] = m.group(2).replace(",", "") if m else None
        r["parse_ok"] = m is not None
        return r
    except Exception as exc:
        return {"url":url, "http_status":None, "error":type(exc).__name__+": "+str(exc), "parse_ok":False}


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--index", default=INDEX)
    args = arg.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        index = retrieve(args.index)
    except Exception as exc:
        print(json.dumps({"index_url": args.index, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
    page = index.pop("_raw").decode("utf-8", "replace")
    paths = sorted(set(re.findall(r'href="(/c/20\d{2}-\d{2}-\d{2}/\d+\.shtml)"[^>]*>【CEA月报】', page)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pages = sorted(pool.map(month_record, paths), key=lambda x:x["url"])
    response = {
        "classification": "OFFICIAL_CEA_MONTHLY_SOURCE_AUDIT_NOT_DAILY_MARKET_SERIES",
        "retrieved_utc":stamp,
        "index": index,
        "listed_2026_monthly_notices":len(paths),
        "official_pages_http200":sum(x.get("status")==200 for x in pages),
        "months_parsed":sum(x.get("parse_ok",False) for x in pages),
        "raw_exchange_pages_publicly_redistributed":False,
        "2026_is_not_confirmatory_for_existing_forecast_study":True,
        "pages":pages,
    }
    print(json.dumps(response,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
