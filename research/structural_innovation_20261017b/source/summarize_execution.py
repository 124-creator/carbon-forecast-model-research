#!/usr/bin/env python3
"""Summarize completed public receipts; never open data, forecasts or weights."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    read = lambda p: json.loads((root / p).read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()
    protocol = read("provenance/PROTOCOL.json")
    selection = read("results/D1_SELECTION.json")
    completed = read("results/D1/COMPLETED.json")
    independent = read("audits/D1_independent/METRIC_AUDIT.json")
    agreement = read("audits/D1_independent/SELECTOR_AGREEMENT.json")
    timing = read("results/TIMING.json")
    configs = read("results/D1_ALL_CONFIGS.json")["rows"]
    jobs = []
    support = {}
    neural_fits = []
    for path in sorted((root / "results/D1/jobs").iterdir()):
        if not path.is_dir():
            continue
        metrics = json.loads((path / "METRICS.json").read_text())
        if metrics["status"] != "completed":
            raise RuntimeError(f"Unexpected incomplete job: {path.name}")
        jobs.append(metrics)
        if metrics["kind"] == "neural":
            neural_fits.append(json.loads((path / "FIT.json").read_text()))
        if metrics["family"] == "smp_full" and metrics["value"] == selection["selected"]["smp_full"]["value"]:
            x = json.loads((path / "SUPPORT.json").read_text())
            support[x["market"]] = {k: x[k] for k in (
                "source_rows", "train_n", "validation_n", "evaluation_n",
                "evaluation_first_date", "evaluation_last_date", "source_sha256",
                "training_state_counts")}
    assert len(jobs) == completed["completed_jobs"] == 160
    assert len(neural_fits) == 120 and len(configs) == 33
    assert independent["status"] == agreement["status"] == "PASS"
    assert selection["advance_to_D2"] is False and agreement["advance_to_D2"] is False
    assert all(sha(p) == expected for p, expected in protocol["source_sha256"].items())
    counts = dict(Counter(x["kind"] for x in jobs))
    assert counts == completed["actual_counts"]
    ledger = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Public completed receipts only; 0 new fits; no CSV, NPZ, weights or confirmation data read",
        "protocol_sha256": sha("provenance/PROTOCOL.json"),
        "frozen_training_sources_verified": len(protocol["source_sha256"]),
        "formal_stage": "D1", "development_year": 2022, "neural_seeds": [17],
        "attempted_jobs": completed["attempted_jobs"], "completed_jobs": len(jobs),
        "failed_jobs": completed["failed_jobs"], "actual_job_counts": counts,
        "actual_market_fits": counts["neural"] + counts["ridge"] + counts["markov_mle"],
        "no_fit_forecast_flows": counts["rw"] + counts["project_reference"],
        "derived_ridge_val_new_fits": 0,
        "formal_neural_executed_epochs": sum(x["executed_epochs"] for x in neural_fits),
        "formal_neural_optimizer_steps": sum(x["optimizer_steps"] for x in neural_fits),
        "neural_selected_epoch_zero": sum(x["selected_epoch"] == 0 for x in neural_fits),
        "neural_stop_reasons": dict(Counter(x["stop_reason"] for x in neural_fits)),
        "d1_wall_seconds": completed["wall_seconds"],
        "raw_development_records": sum(x["source_rows"] for x in support.values()),
        "training_origin_rows_across_series": sum(x["train_n"] for x in support.values()),
        "validation_origin_rows_across_series": sum(x["validation_n"] for x in support.values()),
        "unique_evaluation_origin_rows_across_series": sum(x["evaluation_n"] for x in support.values()),
        "forecast_origin_rows_including_config_repetition": sum(x["n"] for x in jobs),
        "forecast_coordinates_including_config_and_horizon_repetition": sum(x["n"] for x in jobs) * protocol["H"],
        "count_interpretation": "Rows across related series, overlapping horizons and repeated configurations are not IID observations",
        "support": support,
        "all_configuration_rows_including_derived_ridge_val": len(configs),
        "selected_families": len(selection["selected"]),
        "matched_families": len(selection["matched_to_smp"]),
        "separate_synthetic_timing_fits": timing["synthetic_fits"],
        "separate_synthetic_timing_optimizer_steps": timing["total_optimizer_steps"],
        "D2_jobs": 0, "real_bootstrap_replicates": 0,
        "confirmation_model_fits": 0, "confirmation_forecasts": 0,
        "confirmation_model_access_this_round": False,
        "original_market_training_reruns_for_delivery": 0,
        "independent_metric_scalar_comparisons": independent["scalar_comparisons"],
        "independent_max_metric_abs_error": independent["max_metric_abs_error"],
        "selector_agreement_counts": agreement["comparison_counts"],
        "selector_agreement_max_abs_difference": agreement["maximum_numeric_absolute_difference"],
        "input_receipt_sha256": {p: sha(p) for p in (
            "results/D1/COMPLETED.json", "results/D1_SELECTION.json", "results/D1_ALL_CONFIGS.json",
            "audits/D1_independent/METRIC_AUDIT.json", "audits/D1_independent/SELECTOR_AGREEMENT.json")}
    }
    stop = {
        "created_utc": ledger["created_utc"],
        "candidate": "SMP / Signed Markov Propagation",
        "status": "STOP_FIXED_CANDIDATE_AFTER_D1",
        "protocol_sha256": ledger["protocol_sha256"],
        "selected_configuration": selection["selected"]["smp_full"]["config_id"],
        "full_score": selection["selected"]["smp_full"]["score"],
        "same_alpha_iid_score": selection["matched_to_smp"]["smp_iid_state"]["score"],
        "same_alpha_one_step_score": selection["matched_to_smp"]["smp_one_step"]["score"],
        "fixed_best_baseline_family": selection["fixed_best_non_smp_baseline"]["family"],
        "fixed_best_baseline_score": selection["fixed_best_non_smp_baseline"]["score"],
        "D1_gate_checks": selection["gate_checks"], "advance_to_D2": False,
        "independent_metric_audit": "PASS", "independent_selector_agreement": "PASS",
        "D2_jobs": 0, "real_bootstrap_replicates": 0,
        "confirmation_model_fits": 0, "confirmation_forecasts": 0,
        "publication_ready_method_contribution_established": False,
        "interpretation": "The completed bounded experiment rejects advancement of this fixed candidate; it neither proves every structural forecasting model ineffective nor establishes a publishable negative-results paper.",
        "post_result_policy": "Keep all configurations and failed scientific hypotheses. Do not change alpha, seeds, loss, horizons or data to rescue this fixed candidate; do not open confirmation for debugging."
    }
    for name, value in (("EXPERIMENT_LEDGER.json", ledger), ("STOP_DECISION.json", stop)):
        path = root / "results" / name
        if path.exists():
            raise FileExistsError(f"Refuse to overwrite {path}")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: ledger[k] for k in ("completed_jobs", "actual_market_fits", "formal_neural_executed_epochs", "formal_neural_optimizer_steps", "neural_selected_epoch_zero", "raw_development_records", "unique_evaluation_origin_rows_across_series")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
