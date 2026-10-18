#!/usr/bin/env python3
"""Run prepared-evidence verification, scoring/replay, or optional fresh training.

Every execution requires a brand-new output directory outside the source study.
Frozen source files are copied byte-for-byte; no production script is patched.
Use --plan-only to inspect all commands without writes, scoring or training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from verify_evidence_bundle import STUDIES, collect_evidence, digest, new_json, read_json, verify_index


def source_study(root, tag):
    return root / STUDIES[tag]


def keep_for_clone(entry, training):
    roles = set(entry["roles"])
    if "frozen_input_or_code" in roles or "freeze_manifest" in roles:
        return True
    if training:
        return bool(roles & {"scorer", "audit_or_score_helper", "causality_prerequisite", "reproduction_tooling"})
    # Old reference scores/audit reports stay in the source package. Recomputed
    # outputs go to empty destinations inside the independent snapshot.
    return not bool(roles & {"reference_metric", "reference_audit"})


def make_steps(source, snapshot, tags, action, python):
    steps = []
    for tag in tags:
        origin, target = source_study(source, tag), source_study(snapshot, tag)
        freeze = read_json(origin / "PROTOCOL_FREEZE.json")
        if action == "train":
            steps.append({"study": tag, "name": "classical_training", "kind": "command",
                          "argv": [python, str(target / "run_classical_baselines.py"), "--study-dir", str(target),
                                   "--freeze-id", "FREEZE:"+freeze["freeze_id"]]})
            runner = "run_factorized_experiment.py" if tag == "fdmc" else "run_neural_experiment.py"
            steps.append({"study": tag, "name": "neural_training", "kind": "command",
                          "argv": [python, str(target / runner), "--study-dir", str(target)]})
            # Installing this auditor earlier would make the original
            # classical runner refuse its nonempty output directory.
            steps.append({"study": tag, "name": "install_classical_auditor_after_fit", "kind": "copy",
                          "source": str(origin / "local_only/classical/verify_saved_predictions.py"),
                          "destination": str(target / "local_only/classical/verify_saved_predictions.py")})
        if action in {"score", "audit", "train"}:
            steps.append({"study": tag, "name": "independent_formal_scores", "kind": "command",
                          "argv": [python, str(target / "independently_score_experiment.py"), "--study-dir", str(target), "--mode", "full"]})
        if action in {"replay", "audit", "train"}:
            steps.append({"study": tag, "name": "classical_saved_model_replay", "kind": "command",
                          "argv": [python, str(target / "local_only/classical/verify_saved_predictions.py")]})
            if tag == "fdmc":
                steps.append({"study": tag, "name": "63_final_neural_checkpoint_replay", "kind": "command",
                              "argv": [python, str(target / "replay_neural_checkpoints.py")]})
    return steps


def compare_scores(source, snapshot, tags):
    result = {}
    fields = ["primary_equal_cell_mse_ratio_to_rw", "secondary_equal_cell_crps_ratio_to_rw",
              "equal_cell_price_mae", "equal_cell_interval80_coverage", "equal_cell_interval80_width"]
    for tag in tags:
        old = read_json(source_study(source, tag) / "results/independent_metrics/INDEPENDENT_METRICS.json")
        new = read_json(source_study(snapshot, tag) / "results/independent_metrics/INDEPENDENT_METRICS.json")
        a, b = ({row["model"]: row for row in z["primary_aggregate"]} for z in (old, new))
        assert set(a) == set(b)
        comparisons = {}
        for model in a:
            comparisons[model] = {}
            for field in fields:
                difference = abs(a[model][field] - b[model][field])
                assert math.isclose(a[model][field], b[model][field], rel_tol=1e-10, abs_tol=1e-9), (tag, model, field)
                comparisons[model][field] = difference
        assert old["gates"]["decision"] == new["gates"]["decision"]
        result[tag] = {"aggregate_scores_match": True, "absolute_differences": comparisons,
                       "gate_decision": new["gates"]["decision"]}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="Root containing original files plus corrected/fdmc child studies")
    ap.add_argument("--index", type=Path, help="Existing PRIVATE_EVIDENCE_INDEX.json; otherwise discover it or build an in-memory allow-list")
    ap.add_argument("--action", choices=["verify", "score", "replay", "audit", "train"], default="audit")
    ap.add_argument("--studies", nargs="+", choices=["all", *STUDIES], default=["all"])
    ap.add_argument("--output", type=Path, required=True, help="New directory outside the source evidence tree")
    ap.add_argument("--python", default=sys.executable, help="Python executable with requirements-pinned installed")
    ap.add_argument("--plan-only", action="store_true", help="Print the exact command plan; no output writes or model/data execution")
    args = ap.parse_args()
    source, output = args.study_root.resolve(), args.output.resolve()
    if "all" in args.studies and args.studies != ["all"]:
        ap.error("Use --studies all alone, or select individual study names")
    tags = list(STUDIES) if args.studies == ["all"] else list(dict.fromkeys(args.studies))
    if output == source or output.is_relative_to(source):
        ap.error("--output must be outside the source evidence tree")
    if output.exists():
        ap.error("--output already exists; choose a new path to preserve previous evidence")
    snapshot = output / "study_snapshot"
    steps = [] if args.action == "verify" else make_steps(source, snapshot, tags, args.action, args.python)
    plan = {"action": args.action, "selected_studies": tags, "source_study": str(source), "output": str(output),
            "snapshot_layout": STUDIES, "creates_new_training": args.action == "train",
            "uses_prepared_inputs_without_rebuilding_raw_sources": True,
            "source_production_files_modified": False,
            "replay_scope": "classical model replay in each selected study; 63 final neural checkpoint replay for fdmc only",
            "training_counts_if_selected": {tag: {"neural_fits": 105 if tag == "fdmc" else 120,
                                                  "classical_learned_fits": 27} for tag in tags} if args.action == "train" else {},
            "steps": steps}
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    index_path = args.index or source / "reproduction/PRIVATE_EVIDENCE_INDEX.json"
    index = read_json(index_path) if index_path.exists() else collect_evidence(source)
    verification = verify_index(source, index)
    output.mkdir(parents=True, exist_ok=False)
    new_json(output / "PLAN.json", plan)
    new_json(output / "SOURCE_PRIVATE_EVIDENCE_INDEX.json", index)
    new_json(output / "SOURCE_VERIFICATION.json", verification)
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(), "action": args.action,
               "status": "RUNNING", "source_training_artifacts_are_historical": True,
               "new_outputs_are_reproductions_not_original_run_records": True, "steps": []}
    if args.action == "verify":
        receipt.update(status="COMPLETE", finished_utc=datetime.now(timezone.utc).isoformat())
        new_json(output / "REPRODUCTION_RECEIPT.json", receipt)
        print(json.dumps({"status": "PASS", "verified_files": verification["checked_files"], "output": str(output)}))
        return
    for entry in index["files"]:
        if keep_for_clone(entry, args.action == "train"):
            destination = snapshot / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / entry["path"], destination)
            assert digest(destination) == entry["sha256"]
    (output / "logs").mkdir()
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    try:
        for i, step in enumerate(steps):
            print(json.dumps({"event": "step_start", "index": i, "study": step["study"], "name": step["name"]}), flush=True)
            if step["kind"] == "copy":
                destination = Path(step["destination"])
                assert destination.is_relative_to(snapshot) and not destination.exists()
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(step["source"], destination)
            else:
                log = output / "logs" / f"{i:02d}_{step['study']}_{step['name']}.log"
                with log.open("x", encoding="utf-8") as handle:
                    subprocess.run(step["argv"], cwd=snapshot, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, check=True)
            receipt["steps"].append({"index": i, "study": step["study"], "name": step["name"], "status": "COMPLETE"})
        if args.action in {"score", "audit"}:
            new_json(output / "FORMAL_SCORE_REPRODUCTION_COMPARISON.json", compare_scores(source, snapshot, tags))
        # Recheck the immutable evidence source, not the newly generated model
        # files whose serialization/timestamps need not have historical hashes.
        verify_index(source, index)
        receipt.update(status="COMPLETE", finished_utc=datetime.now(timezone.utc).isoformat(),
                       source_evidence_hashes_unchanged=True)
        new_json(output / "REPRODUCTION_RECEIPT.json", receipt)
        print(json.dumps({"status": "COMPLETE", "action": args.action, "output": str(output)}), flush=True)
    except Exception as error:
        receipt.update(status="FAILED", finished_utc=datetime.now(timezone.utc).isoformat(),
                       error_type=type(error).__name__, error=str(error),
                       source_evidence_was_not_a_run_destination=True)
        new_json(output / "REPRODUCTION_FAILURE.json", receipt)
        raise


if __name__ == "__main__":
    main()
