#!/usr/bin/env python3
"""Incremental final export review; run only after the lead finishes final docs.

Preserves the original PUBLIC_SCOPE_PREFLIGHT.json. Reuses its exact audit script
in a fresh temporary directory, then adds newly acquired Tianjin source hashes,
prediction/checkpoint manifest hashes, exact licensed-data identities, and a
delta against the old snapshot. Optional --export-dir / --archive inspect actual
delivery copies as well. No upload, model fit or source/result edit is performed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

PREVIOUS_SHA = "0e854dcdc4617ef8b4c6f8c1e1fadded35961da8e7044fddf5617cf0fc481d36"
POINT_KEYS = {
    "price", "prices", "close", "closing_price", "mean_price", "mean_price_yuan_per_ton",
    "volume", "volumes", "volume_tons", "total_volume_tons", "turnover", "turnover_yuan",
    "total_turnover_yuan", "observed", "observed_change", "predicted_change", "predictions",
    "y_true", "y_pred", "actual_price", "target_prices", "price_path", "test_predictions",
    "forecast_path", "per_origin_loss", "mse_by_origin",
}
STATE_KEYS = {"state_dict", "state_dict_including_reference_buffer"}
RAW_SUFFIXES = {".xlsx", ".xls", ".xlsb", ".npz", ".npy", ".pt", ".pth", ".ckpt",
                ".pkl", ".pickle", ".ipynb", ".db", ".sqlite", ".sqlite3", ".parquet",
                ".feather", ".h5", ".hdf5", ".arrow"}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new receipt; preserve every prior snapshot")
    repo, study = args.repo.resolve(), args.study.resolve()
    prior_path = study / "audits/PUBLIC_SCOPE_PREFLIGHT.json"
    prior_body = prior_path.read_bytes()
    if sha(prior_body) != PREVIOUS_SHA:
        raise ValueError("Original public-scope receipt changed")
    prior = json.loads(prior_body)
    old_script = study / "audits/audit_public_scope.py"
    if sha(old_script.read_bytes()) != prior["audit_script_sha256"]:
        raise ValueError("Original public-scope audit script changed")
    with tempfile.TemporaryDirectory(prefix="carbon-final-scope-base-") as temp:
        base_path = Path(temp) / "BASE_RECHECK.json"
        result = subprocess.run(
            [sys.executable, str(old_script), "--repo", str(repo), "--study", str(study), "--out", str(base_path)],
            capture_output=True, text=True,
        )
        if not base_path.exists():
            raise RuntimeError("Base scope recheck did not complete: " + result.stderr[-2000:])
        base = json.loads(base_path.read_text())
    issues = [{"scope": "study_base", **issue} for issue in base["issues"]]
    if base["files_changed_during_check"]:
        issues.append({"scope": "study_base", "issue": "files_changed_during_base_check",
                       "paths": base["files_changed_during_check"]})

    forbidden = {}
    author = json.loads((study / "provenance/PUBLIC_AUTHOR_DOWNLOAD_CLI_CHECK.json").read_text())
    for row in author["files"]:
        forbidden[row["sha256"]] = "unlicensed author file: " + row["local"]
    for manifest in sorted((study / "results").glob("*/PREDICTION_FILE_MANIFEST.json")) + sorted((study / "results").glob("*/CHECKPOINT_MANIFEST.json")):
        for row in json.loads(manifest.read_text()):
            forbidden[row["sha256"]] = manifest.parent.name + ": " + row["relative_path"]
    private_names = [
        "tianjin_tshq.html", "tianjin_body.js", "tianjin_body_decoded.html", "tianjin_tables.json",
        "TIANJIN_OFFICIAL_DAILY_MEAN_snapshot.csv", "TIANJIN_OFFICIAL_MEAN.csv", "db.sqlite3",
        "bea.py", "tjea.py", "TIANJIN_SCIENCEDB_OVERLAP_MECHANICAL.json",
    ]
    private_root = Path("/tmp/carbon-next-market-data-20261019d")
    private_evidence = []
    for name in private_names:
        path = private_root / name
        if path.is_file():
            body = path.read_bytes()
            digest = sha(body)
            forbidden[digest] = "new Tianjin/author acquisition: " + name
            private_evidence.append({"path": str(path), "bytes": len(body), "sha256": digest})
    for digest, name in {
        "1e36a5c1f19e271adef449b6232602be8b31b504016776e6436a83ea212feb9e": "author notebook",
        "77f93f3c0579d350196975719aeb2e2e12684bf2e946933db74a07854c2afa00": "GZ auxiliary per-date table",
        "15d33910cc9c35d3f51a8043d6d909d5e045aa2c11ee0e55482c2cb5cdcf571d": "HB auxiliary per-date table",
        "d21795622bbb037772ff39f63c79b689ada5ad21721740219ca38efa7c660715": "EWT point predictions",
        "c3e4c075832f6e5087cf416adecbae9d3c1b61859a6ad09fdf84294197393238": "official Tianjin Body.js",
        "85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a": "official Tianjin canonical CSV",
        "868f865f8b4e0ffbcf200f7fe5191d926c9d7eb904eb5e6b977cfab64618ba97": "official Tianjin first CSV",
        "09f08f1d42fdfb41345bfbc57586330a9317b6c1b74ca7ea2caa92545eaef079": "unlicensed author MATLAB locally saved text",
        "9f3214d0ce3314210cc755d26e8d66838773c376a79cb309f8c4bb7289ba37fe": "unlicensed author MATLAB upstream bytes",
    }.items():
        forbidden[digest] = name
    package = json.loads((study / "data/sciencedb_819_v1/provenance/PACKAGE_MANIFEST.json").read_text())
    licensed = {row["sha256"]: row["path"] for row in package["files"]
                if Path(row["path"]).suffix in {".xls", ".csv"}}
    licensed_documents = {row["sha256"]: row["path"] for row in package["files"]}
    snapshots, csv_profiles, state_locations = [], [], []

    def inspect(label, path, body):
        digest = sha(body)
        permitted_daily = digest in licensed
        permitted_source_document = digest in licensed_documents
        suffix = PurePosixPath(path).suffix.lower()
        snapshots.append({"scope": label, "path": path, "bytes": len(body), "sha256": digest})
        def flag(issue, **extra):
            issues.append({"scope": label, "path": path, "issue": issue, **extra})
        if digest in forbidden:
            flag("known_private_data_or_binary_checkpoint", identity=forbidden[digest])
        if suffix in RAW_SUFFIXES and not permitted_daily:
            flag("raw_or_checkpoint_extension_without_exact_licensed_identity")
        if body.startswith(b"SQLite format 3\x00"):
            flag("sqlite_content_even_if_renamed")
        if suffix in {".csv", ".tsv"}:
            reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")), delimiter="\t" if suffix == ".tsv" else ",")
            columns = reader.fieldnames or []
            count = sum(1 for _ in reader)
            csv_profiles.append({"scope": label, "path": path, "columns": columns, "rows": count,
                                 "exact_licensed_daily_data": permitted_daily})
            if not permitted_daily and (set(columns) & {"origin_row", "origin_date", "predicted_change"} or
                                       (set(columns) & {"date", "trade_date"} and set(columns) & POINT_KEYS)):
                flag("point_level_table_outside_exact_licensed_data")
        if suffix == ".json" or path.endswith(".json.gz"):
            obj = json.loads(gzip.decompress(body) if path.endswith(".json.gz") else body)
            def walk(value, pointer=""):
                if isinstance(value, dict):
                    if not permitted_source_document and set(value) & {"date", "trade_date", "origin_date"} and set(value) & POINT_KEYS:
                        flag("per_date_observation_or_forecast_object", json_pointer=pointer)
                    for key, item in value.items():
                        sub = pointer + "/" + key
                        if key in STATE_KEYS:
                            state_locations.append({"scope": label, "path": path, "json_pointer": sub,
                                                    "authorized_own_model_parameters": True})
                            continue
                        if not permitted_source_document and key in POINT_KEYS and isinstance(item, list) and len(item) > 20:
                            flag("long_point_field", json_pointer=sub, length=len(item))
                        walk(item, sub)
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        walk(item, pointer + "/" + str(index))
            walk(obj)

    for entry in base["file_snapshot"]:
        path = study / entry["path"]
        if path.is_symlink():
            issues.append({"scope": "study", "path": entry["path"], "issue": "symlink_requires_explicit_export_review"})
        inspect("study", entry["path"], path.read_bytes())
    extra_files = []
    if args.export_dir:
        for path in sorted(args.export_dir.rglob("*")):
            if path.is_file():
                label = path.relative_to(args.export_dir).as_posix()
                inspect("export_directory", label, path.read_bytes())
                extra_files.append((path, sha(path.read_bytes())))
    archive_sha = None
    if args.archive:
        archive_sha = sha(args.archive.read_bytes())
        with zipfile.ZipFile(args.archive) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if PurePosixPath(member.filename).is_absolute() or ".." in PurePosixPath(member.filename).parts:
                    issues.append({"scope": "archive", "path": member.filename, "issue": "unsafe_archive_path"})
                inspect("archive", member.filename, archive.read(member))

    old_map = {entry["path"]: entry["sha256"] for entry in prior["file_snapshot"]}
    new_map = {entry["path"]: entry["sha256"] for entry in snapshots if entry["scope"] == "study"}
    delta = {"added": sorted(set(new_map) - set(old_map)), "removed": sorted(set(old_map) - set(new_map)),
             "modified": sorted(path for path in set(new_map) & set(old_map) if new_map[path] != old_map[path]),
             "unchanged_count": sum(new_map.get(path) == digest for path, digest in old_map.items())}
    mutations = [path for path, digest in new_map.items() if not (study / path).is_file() or sha((study / path).read_bytes()) != digest]
    mutations += [str(path) for path, digest in extra_files if sha(path.read_bytes()) != digest]
    rel = study.relative_to(repo).as_posix()
    names_after = set(git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "--", rel).splitlines())
    names_after = {str(Path(name).relative_to(rel)) for name in names_after if (repo / name).is_file()}
    if names_after != set(new_map):
        issues.append({"scope": "study", "issue": "file_inventory_changed_during_check",
                       "added": sorted(names_after - set(new_map)), "removed": sorted(set(new_map) - names_after)})
    if sha(prior_path.read_bytes()) != PREVIOUS_SHA:
        issues.append({"scope": "prior_receipt", "issue": "historical_receipt_changed_during_check"})
    if args.archive and sha(args.archive.read_bytes()) != archive_sha:
        mutations.append(str(args.archive))
    receipt = {
        "status": "PASS_FINAL_PUBLIC_EXPORT_SCOPE" if not issues and not mutations else "REVIEW_REQUIRED",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "git_head": git(repo, "rev-parse", "HEAD"), "study": rel,
        "prior_receipt_path": str(prior_path.relative_to(study)), "prior_receipt_sha256": PREVIOUS_SHA,
        "prior_receipt_preserved": True, "base_recheck": base,
        "delta_since_original_public_scope": delta,
        "root_publication_rule": "Own trained candidate/control numeric state_dict JSONs are explicitly public; full binary checkpoints, point predictions and newly acquired unlicensed author/Tianjin raw or per-date files remain private. Exact licensed ScienceDB XLS/CSV identities remain public.",
        "known_private_content_hashes": len(forbidden), "private_acquisition_identity_checks": private_evidence,
        "exact_licensed_data_identities": licensed, "csv_profiles": csv_profiles,
        "authorized_numeric_state_dict_locations": state_locations,
        "archive_sha256": archive_sha, "files_changed_during_check": mutations,
        "issues": issues, "file_snapshot": snapshots,
        "limitations": ["The receipt itself is created after its input snapshot and is not self-hashed.",
                        "Git-ignored runtime files are not proposed Git exports; optional export directory/archive is reviewed directly when supplied.",
                        "Known hashes and schema checks do not prove that arbitrary encoded data cannot hide in text or image pixels.",
                        "Later additions or modifications require another new receipt."],
        "market_fits": 0, "uploads": 0, "source_or_result_edits": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "inspected_files": len(snapshots),
                      "known_private_hashes": len(forbidden), "issues": len(issues),
                      "mutations": len(mutations), "out": str(args.out)}, ensure_ascii=False))
    if issues or mutations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
