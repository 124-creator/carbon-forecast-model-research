#!/usr/bin/env python3
"""Verify/index the private prepared-data evidence bundle using only stdlib.

No model deserialization, training, scoring, source-workbook reading or network.
The manifest is a positive allow-list, not a recursive archive of the workspace.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

STUDIES = {"original": ".", "corrected": "calendar_correction_20261018/rerun",
           "fdmc": "scale_separated_revision"}
FREEZE_SHA = {
    "original": "0030a5fdfaa75399a7845d4d9cfe215b49067e1779078dae553c6cc70b005530",
    "corrected": "a219303d4680f3db5e74709184191d0f02fb16e1ddc02fc8c23a031c962d3db8",
    "fdmc": "327c96fb050e60b86cb58dfcf23c47ca971ddfe910201687cf1c54f7f8917a62",
}
AGGREGATE_FILES = ["INDEPENDENT_METRICS.json", "AGGREGATE.csv", "FULL_CELLS.csv",
                   "PER_SEED_AGGREGATE.csv", "PER_SEED_CELLS.csv", "PAIRED_CI.json",
                   "PAIRED_CI.csv", "SECONDARY_DIRECT_CELL_CI.csv", "GATES.json",
                   "SCORING_CHECKS.json", "PREDICTION_AUDIT.json", "OUTPUT_MANIFEST.json",
                   "INDEPENDENT_REVIEW_ZH.md", "BOOTSTRAP_REPLICATES.npz"]
REPRODUCTION_FILES = ["REPRODUCE_ZH.md", "requirements-pinned.txt", "verify_evidence_bundle.py", "reproduce.py"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def inside(root, path):
    root, path = root.resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Bundle reference escapes the study root: {path}")
    return path.relative_to(root).as_posix()


def collect_evidence(study_root):
    root = Path(study_root).resolve()
    paths, counts, freeze_checks = {}, {}, []

    def add(path, role, tag="shared", required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return
        if path.is_symlink():
            raise ValueError(f"Private evidence must contain regular files: {path}")
        relative = inside(root, path)
        if path.suffix.lower() in {".pdf", ".xls", ".xlsx", ".zip"}:
            raise ValueError(f"Raw workbook/paper/archive is outside this prepared-data allow-list: {relative}")
        entry = paths.setdefault(relative, {"path": relative, "roles": set(), "studies": set()})
        entry["roles"].add(role)
        entry["studies"].add(tag)

    for tag, relative in STUDIES.items():
        study = root / relative
        freeze_file = study / "PROTOCOL_FREEZE.json"
        assert digest(freeze_file) == FREEZE_SHA[tag], (tag, "protocol identity")
        freeze, cfg = read_json(freeze_file), read_json(study / "experiment_config.json")
        assert cfg["evaluation_issue_years"] == [2021, 2022, 2023]
        assert cfg["horizons"] == [1, 5, 20] and cfg["refit_seeds"] == [17, 29, 43]
        add(freeze_file, "freeze_manifest", tag)
        for rel, expected in freeze["frozen_sha256"].items():
            p = study / rel
            assert digest(p) == expected, (tag, rel, "frozen bytes changed")
            add(p, "frozen_input_or_code", tag)
        freeze_checks.append({"study": tag, "freeze_id": freeze["freeze_id"],
                              "freeze_sha256": FREEZE_SHA[tag],
                              "frozen_entries_checked": len(freeze["frozen_sha256"]), "pass": True})
        add(study / "independently_score_experiment.py", "scorer", tag)
        add(study / "local_only/classical/verify_saved_predictions.py", "classical_auditor", tag)
        if tag == "fdmc":
            for name in ["independent_lognormal_scoring.py", "replay_neural_checkpoints.py", "verify_factor_data_causality.py"]:
                add(study / name, "audit_or_score_helper", tag)
            add(study / "results/FACTOR_DATA_CAUSALITY_CHECK.json", "causality_prerequisite", tag)
            add(study / "results/NEURAL_REPLAY_AUDIT.json", "reference_audit", tag)
            replay = read_json(study / "results/NEURAL_REPLAY_AUDIT.json")
            assert replay["status"] == "PASS" and replay["passed_checkpoint_count"] == 63
        run = read_json(study / "results/neural/RUN_SUMMARY.json")
        assert run["status"] == "COMPLETE" and run["failures"] == 0
        assert run["protocol_freeze_sha256"] == FREEZE_SHA[tag]
        add(study / "results/neural/RUN_SUMMARY.json", "completion_receipt", tag)
        add(study / "results/neural/RUN_PROGRESS.json", "completion_receipt", tag)
        stage = Counter(neural_fits=0, neural_refits=0, classical_learned_fits=0,
                        formal_npz=0, validation_npz=0, classical_fold_jobs=0)
        expected_pt, expected_val, expected_formal_neural = set(), set(), set()
        for method in cfg["neural_models"]:
            for year in cfg["evaluation_issue_years"]:
                add(study / f"results/neural/selection_{method}_{year}.json", "selection_receipt", tag)
                fit_ids = []
                for lr in cfg["learning_rates"]:
                    fit_id = f"{method}_{year}_tune_lr{str(lr).replace('.', 'p')}_s{cfg['tuning_seed']}"
                    fit_ids.append(fit_id)
                    name = fit_id + "_validation.npz"
                    add(study / "local_only/neural" / name, "validation_prediction_private", tag)
                    expected_val.add(name)
                    stage["validation_npz"] += 1
                for seed in cfg["refit_seeds"]:
                    fit_ids.append(f"{method}_{year}_refit_s{seed}")
                    name = f"{method}_{year}_s{seed}.npz"
                    add(study / "local_only/neural" / name, "formal_prediction_private", tag)
                    expected_formal_neural.add(name)
                    stage["formal_npz"] += 1
                    stage["neural_refits"] += 1
                for fit_id in fit_ids:
                    checkpoint = study / f"local_only/neural/{fit_id}.pt"
                    receipt = study / f"results/neural/{fit_id}.json"
                    assert digest(checkpoint) == read_json(receipt)["state_sha256"], fit_id
                    add(checkpoint, "neural_checkpoint_private", tag)
                    add(receipt, "fit_receipt", tag)
                    expected_pt.add(checkpoint.name)
                    stage["neural_fits"] += 1
        actual_pt = {p.name for p in (study / "local_only/neural").glob("*.pt")}
        actual_val = {p.name for p in (study / "local_only/neural").glob("*_validation.npz")}
        actual_other_npz = {p.name for p in (study / "local_only/neural").glob("*.npz")} - actual_val
        assert actual_pt == expected_pt and actual_val == expected_val
        assert actual_other_npz == expected_formal_neural
        assert run["fits"] == stage["neural_fits"] and run["refits"] == stage["neural_refits"]
        classical_receipt = read_json(study / "results/classical/CLASSICAL_RUN_RECEIPT.json")
        assert classical_receipt["status"] == "completed" and len(classical_receipt["completed_fold_models"]) == 12
        for name in ["CLASSICAL_RUN_RECEIPT.json", "CLASSICAL_METRICS.json"]:
            add(study / "results/classical" / name, "classical_reference", tag)
        add(study / "results/classical/CLASSICAL_PREDICTION_AUDIT.json", "reference_audit", tag)
        for method in cfg["classical_models"]:
            for year in cfg["evaluation_issue_years"]:
                stem = f"{method}_{year}"
                for suffix, role in [(".npz", "formal_prediction_private"), (".joblib", "classical_checkpoint_private"),
                                     ("_fit_record.json", "training_identity_private")]:
                    add(study / "local_only/classical" / (stem+suffix), role, tag)
                trace_path = study / f"results/classical/{stem}_trace.json"
                trace = read_json(trace_path)
                add(trace_path, "classical_trace", tag)
                if method in ("Ridge", "HistGB"):
                    stage["classical_learned_fits"] += len(trace["candidate_traces"]) + 1
                stage["formal_npz"] += 1
                stage["classical_fold_jobs"] += 1
        for name in AGGREGATE_FILES:
            add(study / "results/independent_metrics" / name, "reference_metric", tag,
                required=name in {"INDEPENDENT_METRICS.json", "AGGREGATE.csv", "GATES.json"})
        metric = read_json(study / "results/independent_metrics/INDEPENDENT_METRICS.json")
        assert metric["overall_status"] == "PASS" and not metric["confirmation_data_read"]
        counts[tag] = dict(stage)
    for name in REPRODUCTION_FILES:
        add(root / "reproduction" / name, "reproduction_tooling")
    total = Counter()
    for stage in counts.values():
        total.update(stage)
    assert total["neural_fits"] == 345 and total["classical_learned_fits"] == 81
    assert total["formal_npz"] == 243 and total["validation_npz"] == 138
    assert total["neural_refits"] == 207 and total["classical_fold_jobs"] == 36
    files = []
    for relative, entry in sorted(paths.items()):
        p = root / relative
        files.append({**entry, "roles": sorted(entry["roles"]), "studies": sorted(entry["studies"]),
                      "bytes": p.stat().st_size, "sha256": digest(p)})
    return {"schema": "carbon-private-prepared-evidence-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "private_only": True, "source_root_is_relocatable": True, "studies": STUDIES,
            "counts_by_study": counts, "counts_total": dict(total), "v2_checkpoints_replayed": 63,
            "frozen_checks": freeze_checks, "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files),
            "files": files,
            "excluded_by_design": ["external paper full texts", "original EEX workbooks/archives", "2024/2025 confirmation data", "workspace caches and environments"],
            "public_boundary": "This is a private allow-list. Prepared price inputs, per-origin predictions and all checkpoints are not default-public Git artifacts. Public source/aggregate publication needs its own filtered manifest.",
            "self_reference_note": "The index and its verification reports are metadata envelope files; they are not hashed recursively inside this index."}


def verify_index(root, index):
    assert index["schema"] == "carbon-private-prepared-evidence-v1"
    checked = set()
    for entry in index["files"]:
        path = Path(root) / entry["path"]
        assert inside(Path(root), path) == entry["path"]
        assert entry["path"] not in checked
        checked.add(entry["path"])
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], entry["path"]
    assert len(checked) == index["file_count"]
    current = collect_evidence(root)
    assert current["counts_total"] == index["counts_total"]
    assert {e["path"] for e in current["files"]} == checked
    return {"status": "PASS", "checked_files": len(checked), "checked_bytes": index["total_bytes"],
            "counts_by_study": current["counts_by_study"], "counts_total": current["counts_total"],
            "v2_checkpoints_replayed_in_original_audit": 63, "all_file_hashes_match": True,
            "all_345_neural_state_hashes_match_fit_receipts": True,
            "all_protocol_freezes_match": True, "model_deserialization_or_training": False,
            "checked_utc": datetime.now(timezone.utc).isoformat()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--index", type=Path, help="Verify an existing private allow-list index")
    ap.add_argument("--write-index", type=Path, help="Create an index at a new path; refuses overwrite")
    ap.add_argument("--report", type=Path, help="Write verification metadata at a new path")
    args = ap.parse_args()
    if args.index and args.write_index:
        ap.error("Choose --index or --write-index")
    root = args.study_root.resolve()
    index = read_json(args.index) if args.index else collect_evidence(root)
    report = verify_index(root, index)
    if args.write_index:
        new_json(args.write_index, index)
    if args.report:
        new_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
