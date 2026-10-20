"""Fetch only public CNEEEX 2021-2023 archive links, retaining source bytes.

This acquisition script does not infer missing prices, run models, or use the
third-party database values. It follows links actually present in archive pages.
All report URLs are date-checked before requests. Four concurrent GETs maximum.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
from prepare_eua_sources import require_non_git_path
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path("/tmp/carbon-cea-official-pre2024")
BASE = "https://overview.cneeex.com"
YEARS = (2021, 2022, 2023)
MAX_WORKERS = 4


def fetch(url, destination):
    """Use the same public GET URL; never attempt to bypass access barriers."""
    if destination.exists():
        data = destination.read_bytes()
        return {
            "url": url, "file": str(destination.relative_to(ROOT)),
            "status": 200, "cached": True, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    time.sleep(0.25)
    started = datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(url, timeout=35) as response:
            data = response.read()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
        destination.write_bytes(data)
        return {
            "url": url, "file": str(destination.relative_to(ROOT)),
            "status": 200, "cached": False, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "downloaded_utc": started,
        }
    except Exception as error:
        return {
            "url": url, "file": str(destination.relative_to(ROOT)),
            "status": "failed", "error": str(error),
            "attempted_utc": started,
        }


def index_url(year, page):
    assert year in YEARS
    return f"{BASE}/qgtpfqjy/mrgk/{year}n/" + (
        "" if page == 1 else f"index_{page}.shtml"
    )


def index_path(year, page):
    return ROOT / (f"index_{year}.html" if page == 1 else f"index_{year}_{page}.html")


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = require_non_git_path(args.output_dir)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    manifest = []
    seen = set()
    frontier = {(year, 1) for year in YEARS}
    reports = {}
    discovered_urls = {(year, 1): index_url(year, 1) for year in YEARS}
    while frontier:
        batch = sorted(frontier - seen)
        frontier = set()
        if not batch:
            break
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(fetch, discovered_urls[(y, n)], index_path(y, n)): (y, n)
                for y, n in batch
            }
            for future in as_completed(futures):
                year, page = futures[future]
                record = future.result()
                manifest.append(record)
                seen.add((year, page))
                if record["status"] != 200:
                    print("INDEX_FAILED", year, page, record["error"], flush=True)
                    continue
                text = index_path(year, page).read_text(encoding="utf-8")
                for target in re.findall(r'href=[\"\']([^\"\']*index_\d+\.shtml)[\"\']', text):
                    number = int(re.search(r"index_(\d+)\.shtml$", target).group(1))
                    if number > 40:
                        raise RuntimeError("Unexpected archive pagination")
                    absolute = urllib.parse.urljoin(record["url"], target)
                    parts = urllib.parse.urlsplit(absolute)
                    if parts.hostname not in ("overview.cneeex.com", "www.cneeex.com"):
                        continue
                    if not (
                        f"/qgtpfqjy/mrgk/{year}n/" in parts.path
                        or re.match(r"/zcms/ui/catalog/\d+/pc/index_\d+\.shtml$", parts.path)
                    ):
                        continue
                    discovered_urls[(year, number)] = absolute
                    frontier.add((year, number))
                for date, ident, title in re.findall(
                    r'<a[^>]*href=[\"\']/c/(\d{4}-\d{2}-\d{2})/(\d+)\.shtml[\"\'][^>]*>(.*?)</a>',
                    text, flags=re.S,
                ):
                    title = html.unescape(re.sub(r"<[^>]*>", "", title)).strip()
                    if int(date[:4]) != year or not date < "2024-01-01":
                        continue
                    if "全国碳市场每日" not in title:
                        continue
                    key = (date, ident)
                    reports[key] = {
                        "date": date, "article_id": ident, "title": title,
                        "url": f"{BASE}/c/{date}/{ident}.shtml",
                    }
        (ROOT / "source_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("INDEX_PROGRESS", len(seen), "pages", len(reports), "reports", flush=True)

    report_rows = [reports[key] for key in sorted(reports)]
    assert all("2021-01-01" <= row["date"] < "2024-01-01" for row in report_rows)
    (ROOT / "report_index.json").write_text(
        json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = {year: sum(row["date"].startswith(str(year)) for row in report_rows) for year in YEARS}
    print("INDEX_COMPLETE", counts, "pages", sorted(seen), flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for row in report_rows:
            assert row["date"] < "2024-01-01"
            destination = ROOT / "reports" / f"{row['date']}_{row['article_id']}.html"
            futures[pool.submit(fetch, row["url"], destination)] = row
        done = 0
        for future in as_completed(futures):
            row = futures[future]
            record = future.result()
            record.update({"date": row["date"], "article_id": row["article_id"]})
            manifest.append(record)
            done += 1
            if record["status"] != 200:
                print("REPORT_FAILED", row["date"], record["error"], flush=True)
            if done % 25 == 0 or done == len(futures):
                (ROOT / "source_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print("REPORT_PROGRESS", done, "/", len(futures), flush=True)
    (ROOT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ACQUISITION_COMPLETE", len(report_rows), "reports", "failures",
          sum(record["status"] != 200 for record in manifest), flush=True)


if __name__ == "__main__":
    main()
