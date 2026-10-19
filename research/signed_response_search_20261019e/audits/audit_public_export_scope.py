#!/usr/bin/env python3
"""Read-only public-scope snapshot; no model imports, fits, or file deletions.

Raw/canonical market records and per-origin arrays/binary checkpoints stay
private. Our own numerical model parameters in JSON are explicitly authorized
for public review. They are counted, never prohibited merely as arrays/weights.
This scanner complements manual review; it does not certify arbitrary encodings.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
BASE = "01433a22ad610e4e8338075dd13381b4304f768c"
OLD_D = "research/literature_aligned_model_search_20261019d"
PRIVATE_SUFFIXES = {".csv", ".xlsx", ".xls", ".js", ".html", ".db", ".sqlite", ".npz", ".npy", ".pt", ".pth", ".pkl", ".zip"}
BANNED_PUBLIC_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".pkl", ".xlsx", ".xls", ".db", ".sqlite"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_files():
    paths = git("ls-files", "-c", "-o", "--exclude-standard", "-z", "--", str(STUDY.relative_to(REPO)))
    return sorted({REPO / x.decode() for x in paths.split(b"\0") if x})


def old_d_check():
    entries = git("ls-tree", "-r", "-z", BASE, "--", OLD_D).split(b"\0")
    bad, identity = [], []
    for entry in entries:
        if not entry:
            continue
        meta, raw_path = entry.split(b"\t")
        _, kind, expected = meta.decode().split()
        path = raw_path.decode()
        file = REPO / path
        if kind != "blob" or not file.is_file():
            bad.append({"path": path, "issue": "missing_or_nonblob"})
            continue
        raw = file.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        identity.append({"path": path, "blob": actual})
        if actual != expected:
            bad.append({"path": path, "expected_blob": expected, "actual_blob": actual})
    new = git("ls-files", "-o", "--exclude-standard", "-z", "--", OLD_D)
    untracked = [p.decode() for p in new.split(b"\0") if p]
    return {"baseline_commit": BASE, "baseline_tree": git("rev-parse", BASE + "^{tree}").decode().strip(),
            "tracked_files_byte_compared": len(identity), "mismatches": bad,
            "nonignored_untracked_old_d_files": untracked,
            "identity_sha256": sha(json.dumps(identity, sort_keys=True).encode()),
            "scope": "Every delivered stage-d blob compared from current bytes; ignored bytecode/cache files are outside delivered content"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New JSON receipt path; existing receipt refused")
    parser.add_argument("--phase", required=True, help="Human-readable snapshot phase")
    parser.add_argument("--private-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("Preserve earlier audit snapshots; choose a new --out path")
    files = [f for f in public_files() if f.resolve() != out]
    before = {str(f.relative_to(STUDY)): sha(f.read_bytes()) for f in files}
    issues, manual, json_review, csv_review = [], [], [], []
    private_files = set()
    dataset_manifest = STUDY / "provenance/RUN_DATASETS.json"
    specs = json.loads(dataset_manifest.read_text())["datasets"]
    for spec in specs:
        private_files.add(Path(spec["file"]).resolve())
    for directory in args.private_root:
        if not directory.is_dir():
            issues.append({"private_root": str(directory), "issue": "requested_private_root_missing"})
            continue
        private_files.update(f.resolve() for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in PRIVATE_SUFFIXES)
    private_hashes = {sha(f.read_bytes()) for f in private_files}
    # Exact canonical-row strings are kept in memory only, never in a receipt.
    canonical_rows = set()
    for spec in specs:
        with Path(spec["file"]).open(newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                canonical_rows.add(",".join(row[k] for k in ("date", "price", "volume_tons", "turnover_yuan")))

    authorized_parameter_scalars = 0
    for file in files:
        rel = str(file.relative_to(STUDY))
        raw = file.read_bytes()
        if sha(raw) in private_hashes:
            issues.append({"path": rel, "issue": "exact_private_source_or_trace_byte_match"})
        if file.suffix.lower() in BANNED_PUBLIC_SUFFIXES:
            issues.append({"path": rel, "issue": "private_binary_or_source_workbook_extension"})
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            if file.suffix.lower() not in {".png", ".pdf", ".jpg", ".jpeg", ".webp"}:
                manual.append({"path": rel, "reason": "nontext_file_requires_explicit_export_review"})
            continue
        match_count = sum(row in text for row in canonical_rows)
        if match_count:
            issues.append({"path": rel, "issue": "literal_canonical_dated_rows", "matched_row_count": match_count})
        if file.suffix == ".csv":
            rows = list(csv.DictReader(text.splitlines()))
            keys = set(rows[0]) if rows else set()
            if "date" in keys and keys.intersection({"price", "close", "volume_tons", "turnover_yuan"}):
                issues.append({"path": rel, "issue": "dated_market_row_csv"})
            if "origin" in keys or "origins" in keys:
                manual.append({"path": rel, "reason": "per_origin_csv_requires_review"})
            csv_review.append({"path": rel, "rows": len(rows), "columns": sorted(keys)})
        if file.suffix != ".json":
            continue
        value = json.loads(text)
        counts = collections.Counter()
        numeric_arrays = []

        def walk(obj, path="", parameters=False):
            nonlocal authorized_parameter_scalars
            counts[type(obj).__name__] += 1
            if parameters and isinstance(obj, (int, float)) and not isinstance(obj, bool):
                authorized_parameter_scalars += 1
            if isinstance(obj, dict):
                keys = {str(k).lower() for k in obj}
                if "date" in keys and keys.intersection({"price", "close", "volume_tons", "turnover_yuan"}):
                    issues.append({"path": rel, "json_path": path, "issue": "dated_market_record_object"})
                for key, item in obj.items():
                    walk(item, path + "/" + str(key), parameters or key == "state_dict")
            elif isinstance(obj, list):
                numeric = bool(obj) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj)
                if numeric:
                    numeric_arrays.append({"json_path": path, "length": len(obj), "authorized_state_dict": parameters})
                if not parameters and len(obj) > 16:
                    key = path.rsplit("/", 1)[-1].lower()
                    if key in {"prices", "close", "volume", "turnover", "predictions", "observed", "targets", "inputs", "y_true", "y_pred"}:
                        manual.append({"path": rel, "json_path": path, "reason": "long_market_or_prediction_array_requires_review"})
                    elif numeric:
                        manual.append({"path": rel, "json_path": path, "reason": "long_numeric_array_outside_state_dict_requires_semantic_review"})
                for i, item in enumerate(obj):
                    walk(item, path + "/" + str(i), parameters)

        walk(value)
        json_review.append({"path": rel, "type_counts": dict(counts), "numeric_arrays": numeric_arrays})

    search_review = []
    for filename, expected in [("CEA_EXPOSURE_SEARCH_ALL_TEXT.json", "path_sha_objects"), ("CEA_EXPOSURE_SEARCH_FILELIST.json", "path_strings")]:
        file = STUDY / "provenance" / filename
        obj = json.loads(file.read_text())
        matches = obj["matched_files"]
        if expected == "path_sha_objects":
            passed = all(isinstance(x, dict) and set(x) == {"path", "sha256"}
                         and isinstance(x["path"], str) and re.fullmatch("[a-f0-9]{64}", x["sha256"]) for x in matches)
        else:
            passed = all(isinstance(x, str) and "\n" not in x for x in matches)
        passed = bool(passed and obj["count"] == len(matches))
        if not passed:
            issues.append({"path": str(file.relative_to(STUDY)), "issue": "search_receipt_contains_unexpected_payload"})
        search_review.append({"path": str(file.relative_to(STUDY)), "sha256": sha(file.read_bytes()),
                              "entries": len(matches), "entry_schema": expected, "pass": passed,
                              "contains_matched_file_contents": False if passed else "not_certified"})

    old = old_d_check()
    if old["mismatches"] or old["nonignored_untracked_old_d_files"]:
        issues.append({"issue": "old_stage_d_changed"})
    after = {str(f.relative_to(STUDY)): sha(f.read_bytes()) for f in files}
    changed = [p for p in before if before[p] != after[p]]
    if changed:
        issues.append({"issue": "content_changed_during_snapshot", "paths": changed})
    receipt = {"status": "PASS" if not issues and not manual else "REVIEW_REQUIRED", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "phase": args.phase, "scope": "Current Git-visible stage-e files, including nonignored new files; receipt itself follows its snapshot",
               "script_sha256": sha(Path(__file__).read_bytes()), "read_only_except_new_receipt": True,
               "public_file_count": len(files), "public_bytes": sum(f.stat().st_size for f in files),
               "extension_counts": dict(collections.Counter(f.suffix.lower() for f in files)),
               "private_files_hashed": len(private_files), "unique_private_content_hashes": len(private_hashes),
               "canonical_rows_checked_for_literal_copy": len(canonical_rows),
               "own_numeric_state_dict_explicitly_authorized": True, "authorized_parameter_scalars_found": authorized_parameter_scalars,
               "parameter_policy": "Do not remove our trained numeric parameters merely because they are arrays or weights; raw sources and per-origin observations/predictions remain private",
               "search_receipt_review": search_review, "json_review": json_review, "csv_review": csv_review,
               "old_d_preservation": old, "issues": issues, "manual_review_candidates": manual,
               "mutations_during_snapshot": changed, "inventory": [{"path": p, "sha256": h} for p, h in before.items()],
               "limitations": ["Heuristic content checks plus explicit manual current-file review; not a universal detector of arbitrary transformed or encoded source data", "Audit current exports again after the experiment; this snapshot cannot approve files created later", "No market scoring, fitting, checkpoint loading or foreign-code execution in this audit"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "out": str(out), "public_file_count": len(files),
                      "issues": issues, "manual_review_candidates": manual, "old_d_files_compared": old["tracked_files_byte_compared"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
