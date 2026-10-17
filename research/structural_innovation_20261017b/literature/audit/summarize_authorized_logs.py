#!/usr/bin/env python3
"""Describe existing offline logs only; never import a trainer or read price arrays.

The only local_only exception is the parent's explicit authorization for the
twenty pretrain_shrink*.json offline-training receipts. No online state,
prediction array, market CSV or confirmation file is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("AUTHORIZED_LOG_DIAGNOSTIC.json"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    study = workspace / "carbon_workspace/research/event_feedback_innovation_20261017"
    offline = sorted((study / "local_only/D1").glob("*/pretrain_shrink*.json"))
    if len(offline) != 20:
        raise ValueError(f"Expected exactly 20 authorized offline logs; found {len(offline)}")
    ancillary = [
        workspace / "scratch/next_design_audit_20261017/BCM_PUBLIC_LOG_DIAGNOSTIC.json",
        workspace / "scratch/design_audit_20261017/EXPANDED_DEVELOPMENT_DESCRIPTION.json",
        study / "source/rrc_core.py",
        study / "source/run_rrc_development.py",
        workspace / "carbon_workspace/research/cross_market_innovation_20261017/d2_source/fit_core.py",
    ]
    allowlist = {p.resolve() for p in offline + ancillary}
    read_manifest = {}

    def read(p: Path) -> bytes:
        p = p.resolve()
        if p not in allowlist or p.suffix.lower() not in {".json", ".py"}:
            raise PermissionError(f"Input outside narrow audit allowlist: {p}")
        if "confirmation" in str(p).lower():
            raise PermissionError("Confirmation paths are forbidden in this audit")
        raw = p.read_bytes()
        read_manifest[str(p.relative_to(workspace))] = digest(raw)
        return raw

    rows = []
    grouped = defaultdict(list)
    for p in offline:
        d = json.loads(read(p))
        tr = d["trace"]
        support = d["support"]
        row = {
            "series": support["market"], "seed": d["seed"], "shrink": d["shrink"],
            "source_log": str(p.relative_to(workspace)),
            "training_windows": len(support["train_origins"]),
            "validation_windows": len(support["validation_origins"]),
            "teacher_parameters": d["teacher_parameters"],
            "student_parameters": d["student_parameters"],
            "selected_epoch": d["selected_epoch"], "epochs_executed": d["epochs"],
            "selected_validation_score_teacher_plus_student": d["validation_score"],
            "initial_and_selected_joint_state_identical": d["initial_state_sha256"] == d["final_state_sha256"],
            "first_post_update_validation_score": tr[0]["validation_score"],
            "minimum_post_update_validation_score": min(x["validation_score"] for x in tr),
            "teacher_objective_first": tr[0]["teacher_objective"],
            "teacher_objective_last": tr[-1]["teacher_objective"],
            "student_objective_first": tr[0]["student_objective"],
            "student_objective_last": tr[-1]["student_objective"],
            "maximum_teacher_total_preclip_gradient": max(x["teacher_gradient"] for x in tr),
            "maximum_student_total_preclip_gradient": max(x["student_gradient"] for x in tr),
            "clipped_teacher_epochs": sum(x["teacher_gradient"] > 1 for x in tr),
            "clipped_student_epochs": sum(x["student_gradient"] > 1 for x in tr),
            "max_train_label": support["max_train_label"],
            "first_validation_origin": support["first_validation_origin"],
            "validation_support_causal": support["max_train_label"] < support["first_validation_origin"],
        }
        if row["initial_and_selected_joint_state_identical"] != (row["selected_epoch"] == 0):
            raise AssertionError("Unexpected epoch0/state-hash relationship")
        if d["epochs"] != len(tr):
            raise AssertionError("Trace length mismatch")
        rows.append(row)
        grouped[(row["series"], row["seed"])].append(row)
    selected = [sorted(v, key=lambda r: (r["selected_validation_score_teacher_plus_student"], -r["shrink"]))[0]
                for _, v in sorted(grouped.items())]
    bcm = json.loads(read(ancillary[0]))
    if any(bcm[k] for k in ["market_price_reads", "weight_array_reads", "confirmation_reads", "market_fits", "optimizer_steps"]):
        raise AssertionError("Upstream BCM diagnostic violates this audit's read boundary")
    clock = json.loads(read(ancillary[1]))
    if clock["numerical_summary_cutoff"] != "2023-12-31":
        raise AssertionError("Unexpected development-only description cutoff")
    clock_rows = [{k: row[k] for k in ["market", "year", "observations", "returns", "zero_price_change_count", "zero_price_change_fraction"]}
                  for row in clock["market_years"] if int(row["year"]) < 2021]
    totals = defaultdict(lambda: {"returns": 0, "zero_price_change_count": 0})
    for row in clock_rows:
        for key in totals[row["market"]]:
            totals[row["market"]][key] += row[key]
    for total in totals.values():
        total["zero_price_change_fraction"] = total["zero_price_change_count"] / total["returns"]
    for p in ancillary[2:]:
        read(p)  # Hash the exact reviewed source; never import it.
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing development logs and source only; post-hoc explanation, not causal attribution or model rescue",
        "python_version": platform.python_version(),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "input_sha256": read_manifest,
        "rrc_all_twenty_offline_fits": rows,
        "rrc_selected_ten_joint_checkpoints": selected,
        "rrc_summary": {
            "candidate_fits": len(rows), "candidate_epoch0": sum(r["selected_epoch"] == 0 for r in rows),
            "selected_fits": len(selected), "selected_epoch0": sum(r["selected_epoch"] == 0 for r in selected),
            "all_teacher_total_objectives_first_to_last_decrease": all(r["teacher_objective_last"] < r["teacher_objective_first"] for r in rows),
            "all_student_total_objectives_first_to_last_decrease": all(r["student_objective_last"] < r["student_objective_first"] for r in rows),
            "all_gradient_clip_counts": sum(r["clipped_teacher_epochs"] + r["clipped_student_epochs"] for r in rows),
            "all_train_validation_boundaries_pass": all(r["validation_support_causal"] for r in rows),
            "any_epoch40_budget_reached": any(r["epochs_executed"] == 40 for r in rows),
            "checkpoint_selection_uses": "teacher plus student validation output, not teacher alone",
        },
        "bcm_development_only_reused_diagnostic": {k: bcm[k] for k in [
            "selected_lambda1_summary", "strength_comparison", "d1_shrink_comparison",
            "selected_development_scores", "selected_initial_states_all_identical",
            "selected_epoch_equal_pairs", "selected_state_identical_pairs", "limitations"]},
        "pre2021_quote_clock_counts_from_existing_public_summary": clock_rows,
        "pre2021_quote_clock_totals_from_existing_public_summary": dict(totals),
        "clock_support_limitations": clock["source_semantics_limitations"],
        "limitations": [
            "Training objective logs include prediction shrink; unpenalized MSE and prediction amplitude are not separately identified.",
            "Logged gradients are total parameter-gradient norms, not encoder-only or mechanism-only gradients.",
            "A selected epoch0 checkpoint does not mean training was not executed.",
            "BCM epoch0 is a training-fitted ridge readout; RRC epoch0 is an exactly zero output head. Do not merge these meanings.",
            "No quote run-length, conditional signed mean or causal cross-market association is estimated here.",
            "The existing pre2021 clock summary covers Guangzhou, Hubei and reported EU only, not Shanghai or EEX.",
        ],
        "activity": {"market_quote_reads": 0, "array_or_weight_reads": 0, "confirmation_reads": 0,
                     "market_fits": 0, "optimizer_steps": 0, "old_file_writes": 0,
                     "authorized_offline_json_log_reads": len(offline)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rrc_summary": receipt["rrc_summary"], "activity": receipt["activity"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
