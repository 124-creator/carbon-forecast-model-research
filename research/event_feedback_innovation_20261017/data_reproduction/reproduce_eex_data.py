"""Create a new isolated EEX research data workspace, with fixed source bytes.

Commands: check; prepare --output NEWDIR (--archive-local ZIP | --download).
No market model is run. Downloads use the single approved public EEX URL only.
Do not redistribute local_only: EEX quotations are not licensed CC BY.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

ARCHIVE_SHA256 = "c3ff6ffaaf348cbd17f61d4726fdbd45a7ead089a0006568a490dd63bb080231"
ARCHIVE_BYTES = 784392
INDEX_NAME = "EEX_REPRODUCTION_SOURCES.json"
COPY_FILES = (
    "acquire_eex_archive.py", "inspect_eex_source.py", "prepare_eex_data.py",
    "check_eex_preparation.py", "verify_eex_preparation.py", "reproduce_eex_data.py",
    "check_eex_reproduction.py", "requirements-eex.txt", "EEX_DATA_METHODS_ZH.md",
    "provenance/EEX_SOURCE_SCOPE.json", "provenance/EEX_SOURCE_ADDENDUM.json",
    "provenance/EEX_PREPARATION_PROTOCOL.json", "provenance/EEX_PREPARATION_SYNTHETIC_FINAL.json",
)
EXPECTED_OUTPUTS = {
    "development/EEX_EUA_T3PA_daily_development.csv",
    "development/EEX_EUA_T3PA_components_development.csv",
    "sealed_confirmation/EEX_EUA_T3PA_daily_confirmation_2024.csv",
    "sealed_confirmation/EEX_EUA_T3PA_components_confirmation_2024.csv",
    "sealed_confirmation/EEX_EUA_T3PA_daily_confirmation_2025.csv",
    "sealed_confirmation/EEX_EUA_T3PA_components_confirmation_2025.csv",
}


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or name != path.as_posix():
        raise ValueError("Unsafe manifest path")
    return path


def source_manifest(base):
    base = base.resolve()
    index_path = base / INDEX_NAME
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("Expected regular source manifest")
    index = json.loads(index_path.read_bytes())
    files = index["files"]
    names = [row["path"] for row in files]
    for name in names:
        safe_relative(name)
    if len(names) != len(set(names)) or set(names) != set(COPY_FILES):
        raise ValueError("Source allowlist differs from approved runtime files")
    outputs = index["expected_prepared_files"]
    for row in outputs:
        safe_relative(row["path"])
    if len(outputs) != len(EXPECTED_OUTPUTS) or {row["path"] for row in outputs} != EXPECTED_OUTPUTS:
        raise ValueError("Prepared output identity allowlist differs")
    for row in files:
        path = base / row["path"]
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base):
            raise ValueError("Unsafe or missing source file")
        blob = path.read_bytes()
        if len(blob) != row["bytes"] or sha256(blob) != row["sha256"]:
            raise ValueError(f"Source integrity failure: {row['path']}")
    return index


def new_destination(path):
    result = path.expanduser().absolute()
    if result.exists() or result.is_symlink():
        raise FileExistsError("Output must be a new directory; no overwrite or restart")
    return result


def verified_archive(path):
    if not path.is_file():
        raise ValueError("Local ZIP is not a regular file")
    blob = path.read_bytes()
    if len(blob) != ARCHIVE_BYTES or sha256(blob) != ARCHIVE_SHA256:
        raise ValueError("Local archive differs from fixed EEX byte identity")
    return blob


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def command(*args):
    subprocess.run([sys.executable, "-B", *map(str, args)], check=True)


def materialize(base, args):
    manifest = source_manifest(base)
    destination = new_destination(args.output)
    archive_blob = verified_archive(args.archive_local) if args.archive_local is not None else None
    plan = {"output": str(destination), "source_files": len(manifest["files"]),
            "source_manifest_sha256": sha256((base / INDEX_NAME).read_bytes()),
            "mode": "fixed_local_archive" if archive_blob is not None else "one_fixed_official_download",
            "archive_sha256": ARCHIVE_SHA256, "market_fit": False,
            "confirmation_numeric_values_displayed": False,
            "raw_or_derived_data_may_be_redistributed": False}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "no_output_created": True, "network_request": False, **plan}, ensure_ascii=False, indent=2))
        return
    destination.mkdir(parents=True, exist_ok=False)
    runtime = destination / "runtime_source"
    runtime.mkdir()
    for row in manifest["files"]:
        target = runtime / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((base / row["path"]).read_bytes())
    (runtime / INDEX_NAME).write_bytes((base / INDEX_NAME).read_bytes())
    (destination / ".gitignore").write_text("local_only/\n__pycache__/\n*.pyc\n", encoding="utf-8")
    private = destination / "local_only"
    private.mkdir()
    (private / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    source = private / "source"
    prepared = private / "prepared"
    write_json(destination / "WORKSPACE_STARTED.json", {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **plan})
    archive_path = source / "eex_primary_auction_2012_2025.zip"
    scope = runtime / "provenance/EEX_SOURCE_SCOPE.json"
    if archive_blob is not None:
        source.mkdir()
        with archive_path.open("xb") as stream:
            stream.write(archive_blob)
        write_json(source / "LOCAL_SOURCE_RECEIPT.json", {"mode": "previously acquired local bytes", "bytes": len(archive_blob),
                                                       "sha256": sha256(archive_blob), "network_request": False})
    else:
        command(runtime / "acquire_eex_archive.py", "--scope", scope, "--output", source)
        verified_archive(archive_path)
    command(runtime / "prepare_eex_data.py", "--archive", archive_path, "--scope", scope,
            "--addendum", runtime / "provenance/EEX_SOURCE_ADDENDUM.json",
            "--protocol", runtime / "provenance/EEX_PREPARATION_PROTOCOL.json", "--output", prepared)
    for row in manifest["expected_prepared_files"]:
        path = prepared / row["path"]
        if sha256(path.read_bytes()) != row["sha256"]:
            raise ValueError("Prepared file differs from the originally verified CSV byte identity")
    command(runtime / "verify_eex_preparation.py", "--archive", archive_path, "--prepared", prepared,
            "--output", destination / "LOCAL_DATA_VERIFICATION.json")
    write_json(destination / "WORKSPACE_COMPLETE.json", {"finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                                       **plan, "completed": True,
                                                       "development": "local_only/prepared/development/EEX_EUA_T3PA_daily_development.csv",
                                                       "confirmation": "local_only/prepared/sealed_confirmation"})
    print(json.dumps({"completed": True, "workspace": str(destination), "market_fit": False,
                      "confirmation_numeric_values_displayed": False}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("check", help="Verify code/protocol allowlist only; no price archive read or network request")
    prep = commands.add_parser("prepare", help="Create a new isolated local data workspace")
    prep.add_argument("--output", required=True, type=Path)
    sources = prep.add_mutually_exclusive_group(required=True)
    sources.add_argument("--archive-local", type=Path)
    sources.add_argument("--download", action="store_true", help="One normal request to the fixed official EEX ZIP URL")
    prep.add_argument("--dry-run", action="store_true", help="Check sources/local archive identity and plan; no directory creation or network")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    if args.action == "check":
        manifest = source_manifest(base)
        print(json.dumps({"passed": True, "source_files": len(manifest["files"]),
                          "source_manifest_sha256": sha256((base / INDEX_NAME).read_bytes()),
                          "archive_read": False, "network_request": False}, indent=2))
    else:
        materialize(base, args)


if __name__ == "__main__":
    main()
