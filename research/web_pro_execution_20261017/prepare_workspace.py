#!/usr/bin/env python3
"""Restore exactly 21 approved inputs and a reviewed source bundle, never fit models.

Use --dry-run to inspect the plan. The output directory must not already exist.
Network mode only requests the exact pinned raw URLs; redirects are refused.
Local mode verifies a previously restored runtime_path tree without any network.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any
import urllib.error
import urllib.request

STAGE = "research/web_pro_execution_20261017"
INDEX = "provenance/WEB_READ_INDEX.json"
INDEX_SHA256 = "f7f9d0e822c2459529cebbfafe7d5f66719fb1da2fbe62bd95811b9e95a1ef9e"
SNAPSHOT = "c78846539e79dc459d3559ae5d0a410ba15fc0d5"
REPOSITORY = "124-creator/carbon-forecast-model-research"
APPROVED_CSV = "data/parsed/figshare_28682348/guangzhou_reported.csv"
APPROVED_CSV_SOURCE = "research/web_pro_handoff_20261017/data/guangzhou_approved_prefix_950.csv"
APPROVED_SHA256 = "1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070"
BUNDLE_MANIFEST = "REPRODUCTION_SOURCES.json"
MAX_SOURCE_BYTES = 2_000_000

# A strict allowlist: no globbing, old receipts, weights, or point predictions.
SOURCE_FILES = (
    "prepare_workspace.py", "check_environment.py", "verify_prepare_workspace.py",
    "REPRODUCE_ZH.md", "requirements.txt", "ENVIRONMENT_LOCK.json",
    "stage_a.py", "STAGE_A_PROTOCOL.md", "stage_b.py", "STAGE_B_PROTOCOL.md",
    "audit/verify_stage_a.py", "audit/AUTHOR_CORE_IDENTITY.json",
    "audit/verify_stage_b.py", "audit/PROCEED_AUTHOR_CORE_IDENTITY.json",
    "vendor/ltsf_linear/NLinear.py", "vendor/ltsf_linear/DLinear.py",
    "vendor/ltsf_linear/LICENSE", "vendor/ltsf_linear/GITHUB_BLOB_SHAS.json",
    "vendor/ltsf_linear/SOURCE_REVIEW_PROVENANCE.json",
    "vendor/ltsf_linear/VENDOR_PROVENANCE.json",
    "vendor/proceed/SOURCE_MANIFEST.json", "vendor/proceed/LICENSE_NOTICE.md",
    "vendor/proceed/README.md", "vendor/proceed/pyproject.toml",
    "vendor/proceed/adapter/proceed.py", "vendor/proceed/adapter/module/base.py",
    "vendor/proceed/adapter/module/generator.py", "vendor/proceed/adapter/module/ssf.py",
    "vendor/proceed/adapter/module/up.py", "vendor/proceed/adapter/module/down_up.py",
    "vendor/proceed/layers/RevIN.py", "vendor/proceed/models/Linear.py",
    "vendor/proceed/models/NLinear.py", "vendor/proceed/models/DLinear.py",
    "vendor/proceed/models/TCN.py", "vendor/proceed/exp/exp_proceed.py",
    "vendor/proceed/data_provider/data_loader.py", "vendor/proceed/run.py",
)
STAGE_A_FROZEN = {
    "stage_a.py": "08cc2baaa85c84af164d6923e121c36bcfeefbb5ad63d85006271ca278cbdf24",
    "STAGE_A_PROTOCOL.md": "d41440a9525b08bdb27164a4d87d02c1b0cfd97aeddfbde904466dfa1d16e991",
    "vendor/ltsf_linear/NLinear.py": "a3d010481dd5e631afc89923b8abbb1b4a27377650ce44cfd577ba927b9f1825",
    "vendor/ltsf_linear/DLinear.py": "6044be51a3bd745ea4523fdc0373af6ed4406d0501f817a06f691764f736b838",
    "vendor/ltsf_linear/LICENSE": "7d0d8193e78eef828028dd99e7e0104fe4e858be03f94b90d84b20e2d0416708",
}


class PreparationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def relative_path(value: str) -> str:
    require(isinstance(value, str) and bool(value), "Empty/non-text relative path")
    require(not PurePosixPath(value).is_absolute(), "Absolute path refused: " + value)
    pieces = value.split("/")
    require(all(piece not in ("", ".", "..") and re.fullmatch(r"[A-Za-z0-9_.-]+", piece)
                for piece in pieces), "Unsafe relative path: " + value)
    return value


def no_symlinks(path: Path) -> Path:
    """Reject symbolic links in existing components, including a dangling leaf."""
    path = Path(os.path.abspath(os.fspath(path)))
    current = Path(path.anchor)
    for piece in path.parts[1:]:
        current = current / piece
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode), "Symbolic link refused: " + str(current))
    return path


def regular_bytes(root: Path, relative: str, size: int | None = None,
                  digest: str | None = None) -> bytes:
    path = no_symlinks(root / relative_path(relative))
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), "Not a regular file: " + str(path))
    require(0 <= info.st_size <= MAX_SOURCE_BYTES, "Unexpected file size: " + str(path))
    # Reject an old full CSV before opening or reading any part of its content.
    if size is not None:
        require(info.st_size == size, "Byte-count mismatch before read: " + str(path))
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb", buffering=0) as stream:
        opened = os.fstat(stream.fileno())
        require((opened.st_dev, opened.st_ino, opened.st_size) ==
                (info.st_dev, info.st_ino, info.st_size), "File changed before read: " + str(path))
        content = stream.read(info.st_size)
        require(os.fstat(stream.fileno()).st_size == info.st_size,
                "File size changed during read: " + str(path))
    require(len(content) == info.st_size, "Short file read: " + str(path))
    if digest is not None:
        require(sha256(content) == digest, "SHA256 mismatch: " + str(path))
    return content


def parse_index(content: bytes) -> dict[str, Any]:
    require(sha256(content) == INDEX_SHA256, "Pinned WEB_READ_INDEX.json SHA256 mismatch")
    index = json.loads(content)
    require(index["file_count"] == 21 and len(index["files"]) == 21,
            "The approved index must contain exactly 21 entries")
    require(index["material_snapshot_commit"] == SNAPSHOT and
            index["approved_data_records"] == 950, "Approved snapshot/data scope differs")
    paths, targets, urls = set(), set(), set()
    market_files = []
    for row in index["files"]:
        path, target = relative_path(row["path"]), relative_path(row["runtime_path"])
        require(row["snapshot_commit"] == SNAPSHOT, "Unpinned entry")
        raw = f"https://raw.githubusercontent.com/{REPOSITORY}/{SNAPSHOT}/{path}"
        github = f"https://github.com/{REPOSITORY}/blob/{SNAPSHOT}/{path}"
        require(row["raw_url"] == raw and row["github_url"] == github,
                "Unexpected network URL for " + path)
        require(type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_SOURCE_BYTES,
                "Invalid approved byte count")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", row["sha256"])), "Invalid SHA256")
        require(path not in paths and target not in targets and raw not in urls,
                "Duplicate approved source/destination/URL")
        paths.add(path)
        targets.add(target)
        urls.add(raw)
        if target.startswith("data/"):
            market_files.append(row)
    require(len(market_files) == 1 and market_files[0]["path"] == APPROVED_CSV_SOURCE
            and market_files[0]["runtime_path"] == APPROVED_CSV
            and market_files[0]["bytes"] == 29204
            and market_files[0]["sha256"] == APPROVED_SHA256,
            "Only the approved 950-row price CSV is allowed")
    return index


def load_bundle(stage_root: Path) -> tuple[dict[str, bytes], bytes]:
    manifest_bytes = regular_bytes(stage_root, BUNDLE_MANIFEST)
    manifest = json.loads(manifest_bytes)
    require(manifest["schema_version"] == 1 and manifest["approved_index_sha256"] == INDEX_SHA256,
            "Invalid reproduction source manifest")
    rows = manifest["files"]
    require(len(rows) == len(SOURCE_FILES) and {row["path"] for row in rows} == set(SOURCE_FILES),
            "Source manifest must match the explicit source-only allowlist")
    bundle = {}
    for row in rows:
        name = relative_path(row["path"])
        content = regular_bytes(stage_root, name, row["bytes"], row["sha256"])
        if name in STAGE_A_FROZEN:
            require(sha256(content) == STAGE_A_FROZEN[name], "Frozen Stage A source changed: " + name)
        bundle[name] = content
    require(manifest["stage_a_frozen_source_sha256"] == STAGE_A_FROZEN,
            "Original Stage A source anchors differ")
    # These hashes anchor the formal Stage B source to its original PRE_RUN record.
    stage_b_names = {name for name in SOURCE_FILES if name.startswith("vendor/proceed/")}
    stage_b_names.update(("stage_b.py", "STAGE_B_PROTOCOL.md"))
    require(set(manifest["stage_b_frozen_source_sha256"]) == stage_b_names,
            "Incomplete formal Stage B source anchors")
    for name, digest in manifest["stage_b_frozen_source_sha256"].items():
        require(name in bundle and sha256(bundle[name]) == digest,
                "Frozen Stage B source changed: " + name)
    return bundle, manifest_bytes


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        raise PreparationError("HTTP redirect refused; only the pinned original URL is allowed: "
                               + request.full_url)


def download_approved(row: dict[str, Any], opener: Any) -> bytes:
    request = urllib.request.Request(row["raw_url"], headers={
        "Accept-Encoding": "identity", "User-Agent": "carbon-approved-workspace/1"})
    with opener.open(request, timeout=30) as response:
        require(response.status == 200 and response.geturl() == row["raw_url"],
                "Unexpected HTTP status/URL: " + row["raw_url"])
        require(response.headers.get("Content-Encoding", "identity") == "identity",
                "Encoded HTTP response refused: " + row["raw_url"])
        length = response.headers.get("Content-Length")
        require(length is not None and length.isdigit() and int(length) == row["bytes"],
                "Missing or mismatched HTTP Content-Length: " + row["raw_url"])
        # Exact read limit; do not request even one byte beyond the approved file.
        content = response.read(row["bytes"])
    require(len(content) == row["bytes"], "HTTP byte-count mismatch: " + row["raw_url"])
    require(sha256(content) == row["sha256"], "HTTP SHA256 mismatch: " + row["raw_url"])
    return content


def validate_prices(content: bytes) -> None:
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    require(rows[0] == ["date", "author_reported_price", "source_excel_row", "source_date_kind"],
            "Approved CSV header differs")
    records = rows[1:]
    require(len(records) == 950 and all(len(row) == 4 for row in records), "Approved CSV shape differs")
    dates = [row[0] for row in records]
    require(dates[0] == "2018-01-03" and dates[-1] == "2021-12-31" and
            all(a < b for a, b in zip(dates, dates[1:])), "Approved CSV date support differs")
    require(all(math.isfinite(float(row[1])) and float(row[1]) > 0 for row in records),
            "Approved prices are nonfinite/nonpositive")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def destination(output: Path, source: Path, approved_local: Path | None) -> Path:
    output = no_symlinks(output)
    require(not output.exists(), "Output already exists; overwriting or resuming is forbidden: " + str(output))
    require(output.parent.is_dir(), "Output parent must already exist: " + str(output.parent))
    require(not output.is_relative_to(source), "Output must be isolated outside --source-dir")
    if approved_local is not None:
        require(not output.is_relative_to(approved_local), "Output must be outside --approved-local-dir")
    return output


def prepare(source: Path, output: Path, approved_local: Path | None = None,
            dry_run: bool = False) -> dict[str, Any]:
    source = no_symlinks(source)
    require(source.is_dir(), "Source repository root does not exist")
    if approved_local is not None:
        approved_local = no_symlinks(approved_local)
        require(approved_local.is_dir(), "Approved local runtime tree does not exist")
    output = destination(output, source, approved_local)
    stage_root = source / STAGE
    index_bytes = regular_bytes(stage_root, INDEX)
    index = parse_index(index_bytes)
    bundle, manifest_bytes = load_bundle(stage_root)
    approved = {}
    if approved_local is not None:
        for row in index["files"]:
            approved[row["runtime_path"]] = regular_bytes(
                approved_local, row["runtime_path"], row["bytes"], row["sha256"])
        validate_prices(approved[APPROVED_CSV])
    plan = {
        "status": "DRY_RUN" if dry_run else "PREPARING", "source_dir": str(source),
        "output": str(output), "approved_mode": "local" if approved_local is not None else "network",
        "approved_index_sha256": INDEX_SHA256, "snapshot_commit": SNAPSHOT,
        "approved_files": [{key: row[key] for key in ("runtime_path", "bytes", "sha256", "raw_url")}
                           for row in index["files"]],
        "approved_file_count": 21, "source_file_count": len(bundle),
        "source_files": list(SOURCE_FILES), "market_records": 950,
        "approved_files_verified": len(approved), "network_requests_performed": 0,
        "market_fits_performed": 0, "existing_stage_a_or_b_results_copied": False,
        "approved_historical_evidence_files_restored": True,
        "point_predictions_copied": False,
    }
    if dry_run:
        return plan
    if approved_local is None:
        opener = urllib.request.build_opener(NoRedirect())
        for row in index["files"]:
            approved[row["runtime_path"]] = download_approved(row, opener)
            plan["network_requests_performed"] += 1
        validate_prices(approved[APPROVED_CSV])
    # Every input is in verified memory before a destination directory is created.
    files = dict(approved)
    for name, content in bundle.items():
        target = STAGE + "/" + name
        require(target not in files, "Source/input destination collision")
        files[target] = content
    source_read_manifest = [{key: row[key] for key in
                             ("path", "runtime_path", "github_url", "sha256", "bytes")}
                            for row in index["files"]]
    files[STAGE + "/" + INDEX] = index_bytes
    files[STAGE + "/" + BUNDLE_MANIFEST] = manifest_bytes
    files["provenance/WEB_READ_INDEX.json"] = index_bytes
    files["provenance/SOURCE_READ_MANIFEST.json"] = json_bytes(source_read_manifest)
    # This new local directory is never intended for publication or automatic staging.
    files[".gitignore"] = b"**/local_only/\n**/__pycache__/\n*.pyc\n.venv/\n"
    for name in files:
        relative_path(name)
        require(not any(parent.as_posix() in files for parent in PurePosixPath(name).parents
                        if parent.as_posix() != "."), "File/directory destination collision")
    output.mkdir(mode=0o700, exist_ok=False)
    try:
        for name, content in files.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
        plan.update(status="COMPLETE", approved_files_verified=21,
                    completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    files_written=len(files), source_manifest_sha256=sha256(manifest_bytes),
                    output_file_sha256={name: sha256(content) for name, content in files.items()})
        with (output / "WORKSPACE_PREPARATION.json").open("xb") as stream:
            stream.write(json_bytes(plan))
    except BaseException as error:
        failure = {"status": "FAILED", "error_type": type(error).__name__, "error": str(error),
                   "market_fits_performed": 0, "resume_or_overwrite_allowed": False}
        try:
            with (output / "WORKSPACE_PREPARATION_FAILED.json").open("xb") as stream:
                stream.write(json_bytes(failure))
        except OSError:
            pass
        raise
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Final repository root; source files are copied only from the explicit allowlist")
    parser.add_argument("--output", required=True, type=Path,
                        help="New isolated directory; its parent must exist; an existing path is refused")
    parser.add_argument("--approved-local-dir", type=Path,
                        help="Optional previously restored runtime_path tree; exact sizes/hashes; no network")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate local sources and print the plan; no network or output-directory writes")
    args = parser.parse_args()
    try:
        result = prepare(args.source_dir, args.output, args.approved_local_dir, args.dry_run)
    except (PreparationError, OSError, KeyError, json.JSONDecodeError, urllib.error.URLError) as error:
        print(json.dumps({"status": "REFUSED_OR_FAILED", "error_type": type(error).__name__,
                          "error": str(error), "market_fits_performed": 0}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
