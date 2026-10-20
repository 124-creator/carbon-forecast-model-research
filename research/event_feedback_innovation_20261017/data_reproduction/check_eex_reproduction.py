"""Bounded source/guard tests; never downloads or parses observed prices."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import reproduce_eex_data as r


def fails(call, error):
    try:
        call()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def run(base):
    checks = []
    manifest = r.source_manifest(base)
    checks.append("complete exact source allowlist and bytes/SHA256 validation")
    for path in ("../outside", "/absolute", "a/../../escape", "a\\escape", "a/./b"):
        fails(lambda path=path: r.safe_relative(path), ValueError)
    checks.append("path traversal and noncanonical manifest paths rejected")
    with tempfile.TemporaryDirectory(prefix="eex_reproduction_synthetic_") as tmp:
        root = Path(tmp)
        existing = root / "existing"
        existing.mkdir()
        marker = existing / "untouched.txt"
        marker.write_text("synthetic preservation sentinel\n", encoding="utf-8")
        before = marker.read_bytes()
        fails(lambda: r.new_destination(existing), FileExistsError)
        assert marker.read_bytes() == before
        checks.append("existing destination refusal preserves old files")
        dangling = root / "dangling"
        dangling.symlink_to(root / "missing_target")
        fails(lambda: r.new_destination(dangling), FileExistsError)
        checks.append("dangling symlink destination rejected")
        badzip = root / "not_market_data.zip"
        badzip.write_bytes(b"PK synthetic wrong byte identity")
        fails(lambda: r.verified_archive(badzip), ValueError)
        checks.append("archive identity mismatch refused before parsing")
        fixture = root / "source_fixture"
        fixture.mkdir()
        for name in (*r.COPY_FILES, r.INDEX_NAME):
            target = fixture / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((base / name).read_bytes())
        (fixture / "requirements-eex.txt").write_text("synthetic changed bytes\n", encoding="utf-8")
        fails(lambda: r.source_manifest(fixture), ValueError)
        checks.append("source code/protocol tampering fails fixed hash validation")
        changed = json.loads((fixture / r.INDEX_NAME).read_bytes())
        changed["files"][0]["path"] = "../outside"
        (fixture / r.INDEX_NAME).write_text(json.dumps(changed), encoding="utf-8")
        fails(lambda: r.source_manifest(fixture), ValueError)
        checks.append("unsafe manifest entry rejected before file access")
    return manifest, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Synthetic receipt already exists")
    base = Path(__file__).resolve().parent
    manifest, checks = run(base)
    receipt = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
               "source_allowlist_files": len(manifest["files"]), "check_count": len(checks), "checks": checks,
               "check_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "source_manifest_sha256": hashlib.sha256((base / r.INDEX_NAME).read_bytes()).hexdigest(),
               "network_requests": 0, "observed_archive_read": False, "confirmation_prices_read": False, "market_fit": False}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
