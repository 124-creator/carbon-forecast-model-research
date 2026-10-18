#!/usr/bin/env python3
"""Create audited public and complete experiment ZIPs without publishing data.

The public allowlist is separate from the complete research-tree archive.
No source outside the specified study, final PDF and Git receipt is archived.
The complete bundle never includes the separate literature-download directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

TEXT = {".py", ".md", ".json", ".csv", ".txt", ".svg"}
PUBLIC_RESULT_NAMES = {
    "AGGREGATE.csv", "FULL_CELLS.csv", "PAIRED_CI.csv", "PAIRED_CI.json",
    "GATES.json", "INDEPENDENT_METRICS.json", "INDEPENDENT_REVIEW_ZH.md",
    "PER_SEED_AGGREGATE.csv", "PER_SEED_CELLS.csv", "SECONDARY_DIRECT_CELL_CI.csv",
}
STAGES = ["", "calendar_correction_20261018/rerun", "scale_separated_revision"]


def file_meta(path, relative):
    data = path.read_bytes()
    return {"path": str(relative), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob_sha": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}


def public_paths(study):
    paths = set()
    for stage in STAGES:
        root = study / stage
        for p in root.iterdir():
            if not p.is_file():
                continue
            if p.suffix in {".py", ".md"} or p.name in {
                ".gitignore",
                "experiment_config.json", "PROTOCOL_FREEZE.json", "EXAMPLE_AUDIT.json",
                "DATA_QUALITY.json", "DELIVERY_SUMMARY.json",
            }:
                paths.add(p)
        metrics = root / "results/independent_metrics"
        paths.update(p for p in metrics.iterdir() if p.is_file() and p.name in PUBLIC_RESULT_NAMES)
    correction = study / "calendar_correction_20261018"
    paths.update(p for p in correction.iterdir() if p.is_file() and
                 (p.suffix in {".py", ".md"} or p.name in {"INPUT_CHANGE_SUMMARY.json", "CALENDAR_CORRECTION_RECEIPT.json"}))
    for dirname in ["docs", "reporting"]:
        paths.update(p for p in (study / dirname).iterdir() if p.is_file() and p.suffix in TEXT | {".png"})
    for name in ["REPRODUCE_ZH.md", "requirements-pinned.txt", "reproduce.py",
                 "verify_evidence_bundle.py", "LAUNCHER_VALIDATION.json"]:
        paths.add(study / "reproduction" / name)
    for name in ["results/OPERATOR_CHECK.json", "results/EXAMPLE_CAUSALITY_CHECK.json",
                 "scale_separated_revision/results/FACTOR_OPERATOR_CHECK.json",
                 "scale_separated_revision/results/FACTOR_DATA_CAUSALITY_CHECK.json"]:
        paths.add(study / name)
    for p in paths:
        if not p.is_file() or p.is_symlink():
            raise ValueError(f"Not a regular release file: {p}")
        rel = p.relative_to(study)
        if "local_only" in rel.parts or "PRIVATE_EVIDENCE" in p.name or (p.suffix not in TEXT | {".png"} and p.name != ".gitignore"):
            raise ValueError(f"Public boundary violation: {rel}")
    return sorted(paths)


def archive(output, study, selected, pdf, github_receipt, mode):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    prefix = "carbon_auction_research_20261018"
    metadata = [file_meta(p, p.relative_to(study)) for p in selected]
    extra = [(pdf, Path("Carbon_Auction_Model_Research_ZH_20261018.pdf")),
             (github_receipt, Path("GITHUB_DELIVERY_RECEIPT.json"))]
    metadata += [file_meta(p, rel) for p, rel in extra]
    manifest = {"schema": "carbon-auction-release-v1", "release_mode": mode,
                "report_date": "2026-10-18", "file_count_excluding_this_manifest": len(metadata),
                "bytes_before_compression_excluding_this_manifest": sum(x["bytes"] for x in metadata),
                "files": metadata, "manifest_self_reference": "Excluded to avoid recursive hashing.",
                "confirmation_scored": False,
                "third_party_paper_fulltexts_included": False,
                "official_original_workbooks_included": False}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in selected:
            z.write(p, f"{prefix}/{p.relative_to(study).as_posix()}")
        for p, rel in extra:
            z.write(p, f"{prefix}/{rel.as_posix()}")
        z.writestr(f"{prefix}/BUNDLE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        for entry in metadata:
            data = z.read(f"{prefix}/{entry['path']}")
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"Archived bytes differ: {entry['path']}")
        if mode == "public_code_and_aggregate_results":
            forbidden = [n for n in z.namelist() if "/local_only/" in n or
                         Path(n).suffix in {".npz", ".pt", ".joblib", ".xlsx", ".xls"} or
                         "PRIVATE_EVIDENCE" in n]
            if forbidden:
                raise ValueError(f"Private artifact found in public archive: {forbidden}")
    return {"file": str(output.resolve()), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "archive_files": len(metadata) + 1, "verification": "ALL_ARCHIVED_BYTES_MATCH_MANIFEST",
            "release_mode": mode}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--github-receipt", type=Path)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    study = args.study_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public = public_paths(study)
    plan = {"study_root": str(study), "public_files": [file_meta(p, p.relative_to(study)) for p in public],
            "public_file_count": len(public), "public_bytes": sum(p.stat().st_size for p in public),
            "git_text_extensions": sorted(TEXT), "git_binary_extensions": [".png"],
            "public_boundary_check": "PASS"}
    (args.output_dir / "PUBLIC_RELEASE_PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    if args.plan_only:
        print(json.dumps({k: v for k, v in plan.items() if k != "public_files"}))
        return
    if not args.pdf or not args.github_receipt:
        ap.error("--pdf and --github-receipt are required for archive creation")
    complete = sorted(p for p in study.rglob("*") if p.is_file() and not p.is_symlink()
                      and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
                      and not p.name.startswith(".DS_Store"))
    if any("literature_fulltext" in p.parts for p in complete):
        raise ValueError("External paper fulltext directory must not be archived")
    outputs = [archive(args.output_dir / "Carbon_Auction_Public_Code_20261018.zip", study, public,
                       args.pdf, args.github_receipt, "public_code_and_aggregate_results"),
               archive(args.output_dir / "Carbon_Auction_Complete_Experiments_20261018.zip", study, complete,
                       args.pdf, args.github_receipt, "complete_private_experiment_evidence")]
    index = json.loads((study / "reproduction/PRIVATE_EVIDENCE_INDEX.json").read_text())
    all_rel = {p.relative_to(study).as_posix() for p in complete}
    missing = [x["path"] for x in index["files"] if x["path"] not in all_rel]
    if missing:
        raise ValueError(f"Private evidence missing from complete ZIP: {missing}")
    receipt = {"status": "PASS", "outputs": outputs,
               "private_positive_index_files_covered": len(index["files"]),
               "public_plan": str(args.output_dir / "PUBLIC_RELEASE_PLAN.json"),
               "confirmation_data_included": False,
               "official_original_workbooks_or_research_paper_fulltexts_included": False}
    (args.output_dir / "PACKAGE_DELIVERY_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
