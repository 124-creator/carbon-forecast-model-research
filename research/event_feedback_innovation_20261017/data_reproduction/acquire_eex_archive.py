"""One normal HTTPS request for the source archive after exact scope verification.

Private research inputs only. No redirects, retries, credential use, data parsing,
market statistics, model fitting, or publication is performed by this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.request

SCOPE_SHA256 = "05fef41b2a591f0570cef37f1aa34b24e48c7d9a105eb819a1603275a88b650f"
APPROVED_URL = "https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/emission-spot-primary-market-auction-report-2012-2025-data.zip"
EXPECTED_BYTES_FROM_HEAD = 784392


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Unexpected redirect; no replacement source requested")


def write_json_new(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_scope = args.scope.read_bytes()
    if hashlib.sha256(source_scope).hexdigest() != SCOPE_SHA256:
        raise ValueError("Source scope identity mismatch")
    scope = json.loads(source_scope)
    if scope["approved_url"] != APPROVED_URL:
        raise ValueError("Unapproved source URL")
    path = args.output.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to reuse an acquisition directory or restart an attempt")
    path.mkdir(parents=True, exist_ok=False)
    (path / "EEX_SOURCE_SCOPE.json").write_bytes(source_scope)
    receipt = dict(requested_utc=utc(), scope_created_utc=scope["created_utc"],
                   scope_sha256=SCOPE_SHA256,
                   acquisition_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_url=APPROVED_URL, expected_bytes_from_prior_head=EXPECTED_BYTES_FROM_HEAD,
                   retries=0, redirects_followed=0, model_fit=False,
                   publication=False, public_raw_data_redistribution_authorized=False)
    write_json_new(path / "ACQUISITION_STARTED.json", receipt)
    try:
        request = urllib.request.Request(APPROVED_URL, headers={"Accept": "application/zip"})
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            receipt["http_status"] = response.status
            blob = response.read(2 * 1024 * 1024 + 1)
        receipt.update(received_bytes=len(blob), received_sha256=hashlib.sha256(blob).hexdigest())
        if len(blob) != EXPECTED_BYTES_FROM_HEAD:
            raise ValueError("Archive length differs from frozen metadata; do not parse")
        if not blob.startswith(b"PK\x03\x04"):
            raise ValueError("Expected ZIP magic")
        with (path / "eex_primary_auction_2012_2025.zip").open("xb") as stream:
            stream.write(blob)
        receipt.update(ok=True, archive_file="eex_primary_auction_2012_2025.zip",
                       prices_parsed=False, confirmation_prices_displayed=False)
    except Exception as exc:
        receipt.update(ok=False, error_type=type(exc).__name__, error=str(exc))
    receipt["finished_utc"] = utc()
    write_json_new(path / "ACQUISITION_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
