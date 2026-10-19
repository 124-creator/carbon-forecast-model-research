"""Post-exposure, fixed max320 budget sensitivity; no edits to original study.

The only fitting change is FitConfig(maximum_epochs=320). Every originally
nonlinear arm is retained for both markets. This is development evidence,
including CEA, and never replaces the original max80 confirmatory comparison.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import run_signed_response_search as core
from signed_response_models import FitConfig, SignedResponse, fit_response, solve_readout

SOURCE = Path(__file__).resolve().parent
STUDY = SOURCE.parent
FAMILIES = tuple(f for f in core.RESPONSE_FAMILIES if f != "fixed_response")
MAX_EPOCHS = 320


def source_hashes():
    names = ["signed_response_models.py", "run_signed_response_search.py", "run_neural_references.py", Path(__file__).name]
    return {name: core.digest(SOURCE/name) for name in names}


def checked_addendum(path, expected_sha, original_out):
    assert core.digest(path) == expected_sha, "Budget addendum identity changed"
    plan = json.loads(Path(path).read_text())
    assert plan["source_sha256"] == source_hashes(), "Frozen source identity changed"
    assert plan["original_core_protocol_sha256"] == core.digest(original_out/"PROTOCOL_BEFORE_RUN.json")
    completion = json.loads((original_out/"EXECUTION_COMPLETE.json").read_text())
    assert completion["status"] == "completed" and completion["source_hashes_unchanged"] and completion["plan_hash_unchanged"]
    assert plan["families"] == list(FAMILIES) and plan["maximum_epochs"] == MAX_EPOCHS
    assert plan["learning_rates"] == list(core.LRS) and plan["mean_ridge_alphas"] == list(core.MEAN_ALPHAS)
    assert FitConfig().maximum_epochs == 80
    assert (FitConfig().minimum_epochs, FitConfig().patience) == (15, 12)
    core.check_old_sources()
    return plan


def run_job(spec, family, out):
    torch.set_num_threads(1)
    d = core.prepare(spec)
    dest = Path(out)/f"{spec['market']}_{core.FOLD}"/family
    m = d["masks"]
    x, (y, variance) = core.tensor(d["x"]), core.normalized_labels(d)
    trials = []
    for alpha in core.MEAN_ALPHAS:
        for lr in core.LRS:
            config = FitConfig(learning_rate=lr, alpha=alpha, maximum_epochs=MAX_EPOCHS)
            _, receipt = fit_response(family, x[m["train"]], y[m["train"]], variance, config,
                                      x_val=x[m["val"]], y_val=y[m["val"]])
            name = f"selection_alpha{alpha:g}_lr{lr}"
            core.write_json(dest/(name+".json"), receipt)
            torch.save({"family": family, "state_dict": receipt["state_dict"], "selected_epoch": receipt["selected_epoch"]}, dest/(name+".pt"))
            trials.append({"alpha": alpha, "learning_rate": lr,
                           "selected_epoch": receipt["selected_epoch"],
                           "validation_loss": receipt["validation_loss"], "receipt": name+".json",
                           "epochs_executed": receipt["epochs_executed"],
                           "positive_gradient_steps": receipt["positive_gradient_steps"]})
    choice = min(trials, key=lambda r: (r["validation_loss"], r["alpha"], r["learning_rate"], r["selected_epoch"]))
    config = FitConfig(learning_rate=choice["learning_rate"], alpha=choice["alpha"], maximum_epochs=MAX_EPOCHS)
    model, receipt = fit_response(family, x[m["final"]], y[m["final"]], variance, config,
                                  fixed_epochs=choice["selected_epoch"])
    core.write_json(dest/"SELECTION.json", {"selected": choice, "trials": trials,
                       "analysis_role": "post-exposure max320 development sensitivity",
                       "alpha_unit": "unchanged mean weighted squared error plus standardized-column ridge",
                       "epoch0": "unchanged fixed initial dynamics with exact training-only ridge head"})
    core.write_json(dest/"REFIT.json", receipt)
    torch.save({"family": family, "state_dict": model.state_dict()}, dest/"refit.pt")
    with torch.no_grad():
        pred = model(x[m["test"]]).numpy()*d["scale"]
        initial = SignedResponse(family)
        beta, _, _, _, scales = solve_readout(initial.basis(x[m["final"]]), y[m["final"]], variance, choice["alpha"])
        initial.set_coefficients(beta, scales)
        init_pred = initial(x[m["test"]]).numpy()*d["scale"]
    np.savez_compressed(dest/"predictions.npz", predictions=pred[None],
                        fixed_dynamics_refit_readout_predictions=init_pred[None],
                        observed=d["y"][m["test"]], origins=d["origins"][m["test"]],
                        horizons=np.asarray(core.HORIZONS), deterministic=np.array([True]))
    result = core.summarize(d, family, pred)
    result.update({"analysis_role": "post-exposure max320 development sensitivity",
                   "original_role": spec["original_role"],
                   "parameter_counts": model.parameter_counts(), "maximum_epochs": MAX_EPOCHS,
                   "selected_epoch": choice["selected_epoch"],
                   "selection_positive_gradient_steps": sum(t["positive_gradient_steps"] for t in trials),
                   "refit_positive_gradient_steps": receipt["positive_gradient_steps"]})
    assert result["role"] == "development"
    core.write_json(dest/"SUMMARY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-core-out", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--expected-addendum-sha", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-report", type=Path)
    args = parser.parse_args()
    plan = checked_addendum(args.addendum, args.expected_addendum_sha, args.original_core_out)
    original = json.loads((args.original_core_out/"PROTOCOL_BEFORE_RUN.json").read_text())
    specs = copy.deepcopy(original["datasets"])
    for spec in specs:
        spec["original_role"] = spec["role"]
        spec["role"] = "development"
        spec["budget_sensitivity_role"] = "Both original stage-e test tails already scored before this addendum"
    assert [s["market"] for s in specs] == plan["markets"]
    if args.preflight_report:
        if args.preflight_report.exists():
            raise FileExistsError("Use a fresh preflight receipt")
        # Identity/configuration checks only; do not instantiate any fit or
        # calculate original or supplementary validation/test forecast loss.
        manifests = [core.prepare(s)["manifest"] for s in specs]
        core.write_json(args.preflight_report, {"status": "PASS_NO_FIT_NO_SCORE", "market_fits": 0,
                         "new_forecast_scores": 0, "source_sha256": source_hashes(),
                         "addendum_sha256": args.expected_addendum_sha, "input_manifests": manifests,
                         "new_fit_config": FitConfig(maximum_epochs=MAX_EPOCHS).__dict__,
                         "planned_selection_fits": 160, "planned_final_refits": 10})
        print("PASS_NO_FIT_NO_SCORE", flush=True)
        return
    if args.out is None:
        parser.error("Run requires --out")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Budget sensitivity requires a new empty directory")
    protocol = {**plan, "created_unix": time.time(), "datasets": specs,
                "addendum_path": str(args.addendum.resolve()), "addendum_sha256": args.expected_addendum_sha,
                "original_core_out": str(args.original_core_out.resolve()),
                "runtime": core.runtime(),
                "fit_configuration": "Original fit_response; only maximum_epochs changes from80 to320",
                "original_controls": "No refit of RW, price Ridge, interaction Ridge, fixed_response, legacy_fixed_response or neural references. Their original artifacts remain separately identified.",
                "claim_scope": "Every supplementary score is post-exposure development sensitivity; never pooled into original confirmatory family"}
    core.write_json(args.out/"PROTOCOL_BEFORE_RUN.json", protocol)
    for spec in specs:
        core.write_json(args.out/f"{spec['market']}_{core.FOLD}"/"INPUT_MANIFEST.json", core.prepare(spec)["manifest"])
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(run_job, spec, family, str(args.out)): f"{spec['market']}/{family}"
                 for spec in specs for family in FAMILIES}
        for future in as_completed(tasks):
            result = future.result()
            summaries.append(result)
            core.write_json(args.out/"PROGRESS_SUMMARIES.json", summaries)
            print(json.dumps({"completed": tasks[future], "development_sensitivity": True,
                              "mean_mse_over_rw": result["mean_mse_over_rw"],
                              "selected_epoch": result["selected_epoch"]}), flush=True)
    core.write_json(args.out/"ALL_SUMMARIES.json", summaries)
    unchanged = source_hashes() == protocol["source_sha256"]
    plan_unchanged = core.digest(args.addendum) == args.expected_addendum_sha
    core.write_json(args.out/"EXECUTION_COMPLETE.json", {"status": "completed" if unchanged and plan_unchanged else "IDENTITY_FAILURE",
                    "summary_rows": len(summaries), "source_hashes_unchanged": unchanged,
                    "plan_hash_unchanged": plan_unchanged, "development_sensitivity_only": True})
    assert unchanged and plan_unchanged


if __name__ == "__main__":
    main()
