"""Bounded ordinary GET for the explicitly requested public channel-data study.

No authentication, cookies, browser emulation, identity change or fitting.
No retry after an HTTP response; one explicit same-identity resume is permitted
only when a local EPERM denial occurred before any HTTP response.
Raw bytes and request receipts remain under the supplied private temporary root.
The caller must obtain each additional URL from the page or referenced script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--purpose", choices=("discovery", "panel"), default="discovery")
    p.add_argument("--source-url", required=True, help="Observed parent URL or original authorized entry")
    p.add_argument("--evidence", required=True, help="Where this literal URL and any query parameters were observed")
    p.add_argument("--resume-after-local-denial", action="store_true",
                   help="One explicit same-identity retry only after a local EPERM failure before any HTTP response")
    args = p.parse_args()
    parsed = urllib.parse.urlsplit(args.url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("Only ordinary credential-free HTTPS URLs are allowed")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise ValueError("Use a simple raw filename")
    args.out_root.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out_root/"REQUEST_LEDGER.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    if any(r.get("access_wall") and urllib.parse.urlsplit(r["url"]).netloc == parsed.netloc for r in ledger):
        raise RuntimeError("A previous request met an access wall at this host; do not retry or change identity")
    if args.purpose == "discovery" and sum(r["purpose"] == "discovery" for r in ledger) >= 10:
        raise RuntimeError("The authorized discovery budget of ten requests is exhausted")
    raw = args.out_root/args.name
    previous = [r for r in ledger if r["url"] == args.url]
    local_resume = (args.resume_after_local_denial and len(previous) == 1
                    and previous[0]["http_status"] is None and previous[0]["bytes"] == 0
                    and "Operation not permitted" in (previous[0]["error"] or ""))
    if raw.exists() or (previous and not local_resume):
        raise FileExistsError("Do not overwrite or repeat a completed/blocked request")
    started = now()
    error, headers, final_url, body, status = None, {}, args.url, b"", None
    try:
        # Default Python User-Agent is retained throughout; no browser headers,
        # identity switching, credentials, cookies or automated retries.
        with urllib.request.urlopen(args.url, timeout=25) as response:
            status = response.status
            final_url = response.url
            headers = {k: response.headers.get(k) for k in
                       ("Content-Type", "Content-Length", "Last-Modified", "ETag") if response.headers.get(k)}
            body = response.read(32*1024*1024+1)
            if len(body) > 32*1024*1024:
                raise ValueError("Response exceeds the bounded32MiB resource limit")
    except urllib.error.HTTPError as exc:
        status, final_url = exc.code, exc.url
        headers = {k: exc.headers.get(k) for k in ("Content-Type", "Content-Length") if exc.headers.get(k)}
        body = exc.read(2*1024*1024)
        error = f"HTTPError:{exc.code}"
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    raw.write_bytes(body)
    text = body.decode("utf-8", errors="replace")
    wall_markers = [s for s in ("Access Denied", "captcha", "hcaptcha", "cf-chl-", "verify you are human", "验证您的浏览器")
                    if s.lower() in text.lower()]
    access_wall = status in (401, 403, 407, 429, 503) or bool(wall_markers)
    receipt = {"request_number": len(ledger)+1, "purpose": args.purpose,
               "url": args.url, "source_url": args.source_url, "discovery_evidence": args.evidence,
               "method": "GET", "request_identity": "default Python urllib; no credentials/cookies/custom headers/automatic retries",
               "explicit_resume_after_local_eperm": bool(local_resume),
               "started_utc": started, "finished_utc": now(), "http_status": status,
               "final_url": final_url, "response_headers": headers, "error": error,
               "raw_path": str(raw.resolve()), "bytes": len(body),
               "sha256": hashlib.sha256(body).hexdigest(), "access_wall": access_wall,
               "wall_markers": wall_markers, "market_fits": 0, "forecast_scores": 0}
    ledger.append(receipt)
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if error or access_wall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
