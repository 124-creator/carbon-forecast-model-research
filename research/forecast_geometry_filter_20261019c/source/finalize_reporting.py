#!/usr/bin/env python3
"""Post-result reporting only: verify frozen evidence and compute oracle headroom.

This script does not fit, select, or rerun any model. It preserves the formal
SUMMARY.json and writes separate final audit / decision / diagnostic files.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="New reporting directory; never overwrite historical receipts.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((ROOT / "provenance/SOURCE_FREEZE.json").read_text())
    assert all(digest(ROOT / name) == sha for name, sha in freeze["sha256"].items())
    summary_path = ROOT / "results/synthetic/SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    complete = json.loads((ROOT / "results/synthetic/COMPLETE.json").read_text())
    assert complete["complete"] and complete["source_hashes_unchanged"]
    assert digest(summary_path) == complete["summary_sha256"]
    trace_checks = []
    for item in complete["trace_files"]:
        trace = args.traces / item["trace_file"]
        assert trace.stat().st_size == item["trace_bytes"]
        assert digest(trace) == item["trace_sha256"]
        trace_checks.append(dict(item, verified=True))
    audit_names = [
        "INDEPENDENT_CORE_AUDIT_RECOVERY.json",
        "RUNNER_PREFIT_AUDIT_RECOVERY.json",
        "RECOVERY_PIPELINE_AUDIT.json",
        "INDEPENDENT_TRACE_REPLAY.json",
        "INDEPENDENT_PREFIX_REPLAY.json",
    ]
    audits = {}
    for name in audit_names:
        p = ROOT / "audits" / name
        d = json.loads(p.read_text())
        assert d["status"] == "PASS"
        audits[name] = {"status": "PASS", "sha256": digest(p)}
    records = list(csv.DictReader((ROOT / "results/synthetic/SELECTED_RESULTS.csv").open()))
    lookup = {(r["scenario"], int(r["seed"]), r["family"]): r for r in records}
    positive = ["rare_large_shocks", "frequent_moderate_shocks"]
    seed_values = {}
    selections = []
    for scenario in positive:
        vals = []
        for seed in range(1000, 1020):
            candidate = lookup[scenario, seed, "future_cdf"]
            selected = candidate["validation_selected_competitor"]
            control = lookup[scenario, seed, selected]
            oracle = lookup[scenario, seed, "true_oracle"]
            ratio = float(oracle["expected_primary"]) / float(control["expected_primary"]) - 1
            vals.append(ratio)
            selections.append({
                "scenario": scenario, "seed": seed, "selected_control": selected,
                "oracle_expected_primary": float(oracle["expected_primary"]),
                "control_expected_primary": float(control["expected_primary"]),
                "oracle_relative_minus_one": ratio,
            })
        seed_values[scenario] = vals
    combined = np.mean(np.asarray([seed_values[s] for s in positive]), axis=0)
    means = {s: float(np.mean(seed_values[s])) for s in positive}
    means["positive_equal_weight"] = float(np.mean(combined))
    assert means["positive_equal_weight"] > -0.01
    headroom = {
        "status": "POST_RESULT_ANALYTIC_DIAGNOSTIC",
        "not_part_of_original_primary_gate": True,
        "no_new_simulation_or_fit_or_selection": True,
        "metric": "oracle_expected_primary / validation_selected_control_expected_primary - 1",
        "units": "fraction; multiply by 100 for percent",
        "scenario_means": means,
        "positive_paired_seed_values": combined.tolist(),
        "cells": selections,
        "original_required_gain_fraction": 0.01,
        "max_mean_gain_on_these_histories_same_information": -means["positive_equal_weight"],
        "original_one_percent_gate_oracle_attainable_on_these_histories": False,
        "scope": "Fixed realized histories, same observed-history information, same selected controls and mean-of-ratios aggregation; not a universal population bound.",
        "interpretation": "Original total-CRPS resource gate did not account for irreducible score. Preserve it; do not treat failing this gate alone as proof of model worthlessness.",
        "source_csv_sha256": digest(ROOT / "results/synthetic/SELECTED_RESULTS.csv"),
    }
    primary = next(c for c in summary["comparisons"] if
        c["stratum"] == "positive_scenarios_equal_weight" and
        c["control"] == "validation_selected_competitor" and
        c["metric"] == "expected_primary")
    empirical = next(c for c in summary["comparisons"] if
        c["stratum"] == "positive_scenarios_equal_weight" and
        c["control"] == "validation_selected_competitor" and
        c["metric"] == "empirical_primary")
    gate = summary["decision"]
    assert gate["decision"] == "STOP_FIXED_GEOMETRY_CANDIDATE"
    assert sum(c["passed"] for c in gate["checks"].values()) == 4
    final = {
        "study": summary["study"],
        "decision": gate["decision"],
        "original_gate_unchanged": True,
        "original_gate": gate,
        "postrun_independent_audits_complete": True,
        "passed_gate_count": 4, "failed_gate_count": 3,
        "expected_primary_comparison": primary,
        "empirical_primary_comparison": empirical,
        "oracle_headroom_posthoc": {
            "file": "ORACLE_HEADROOM_POSTHOC.json",
            "maximum_gain_fraction": -means["positive_equal_weight"],
            "one_percent_gate_design_limitation_disclosed": True,
            "gate_not_revised": True,
        },
        "stop_evidence_independent_of_one_percent_gate": [
            "Expected CRPS is worse than the validation-selected control; nominal paired-seed interval is above zero.",
            "Full future CDF does not beat future-variance geometry or tempered Bayes.",
            "Empirical mean improvement condition fails; its interval crosses zero.",
        ],
        "market_continuation_authorized_by_gate": False,
        "publication_ready": False,
        "novelty_certified": False,
        "market_validity_established": False,
        "actual_neural_fits": 0,
        "actual_emission_or_transition_parameter_fits": 0,
        "new_market_data_or_confirmation_evaluations": 0,
    }
    audit = {
        "status": "PASS_EXECUTION_AND_EVIDENCE_COMPLETE_NOT_MODEL_GATE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_only": True,
        "frozen_files_unchanged": len(freeze["sha256"]),
        "protocol_sha256": digest(ROOT / "provenance/SYNTHETIC_PROTOCOL.json"),
        "production_core_sha256": digest(ROOT / "source/geometry_filter.py"),
        "production_runner_sha256": digest(ROOT / "source/run_synthetic.py"),
        "source_freeze_sha256": digest(ROOT / "provenance/SOURCE_FREEZE.json"),
        "formal_summary_sha256": digest(summary_path),
        "audit_receipts": audits,
        "verified_generated_trace_files": trace_checks,
        "formal_validation_configurations": summary["validation_filter_trials"],
        "selected_family_streams": summary["selected_family_streams"],
        "aggregated_result_rows": len(records),
        "independent_seeds": summary["independent_random_seeds"],
        "formal_scenario_seed_sequences": summary["unique_simulated_scenario_seed_sequences"],
        "synthetic_observations": summary["synthetic_observations"],
        "actual_neural_fits": summary["actual_neural_fits"],
        "actual_emission_or_transition_parameter_fits": summary["actual_emission_or_transition_parameter_fits"],
        "formal_decision": gate["decision"],
        "market_continuation": False,
        "no_market_data_read_by_this_script": True,
        "publication_claim": False,
        "reporting_script_sha256": digest(Path(__file__)),
        "limits": [
            "Evidence checks and audit replays are not new independent scientific repetitions.",
            "Local freezes are not external preregistration.",
            "Prefix replay reused production state updates; independent test audit used original-objective KKT and independent score laws.",
            "Oracle headroom diagnostic is post-result and does not replace the frozen gate.",
        ],
    }
    for name, value in [
        ("ORACLE_HEADROOM_POSTHOC.json", headroom),
        ("FINAL_DECISION.json", final),
        ("FINAL_EXECUTION_AUDIT.json", audit),
    ]:
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": audit["status"], "decision": gate["decision"],
        "headroom_percent": 100 * -means["positive_equal_weight"],
        "output": str(args.out),
    }))

if __name__ == "__main__":
    main()

