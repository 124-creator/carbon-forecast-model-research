"""Read-only post-result v4 learning/initialization diagnostic for six arms.

No fitting, optimizer step, candidate selection or significance calculation.
Saved selection checkpoints are distinguished from pre-restore optimization
activity. Compact initial parameter summaries avoid duplicating state_dicts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY/"source"))
import run_official_tianjin_extension as official  # noqa: E402
import diagnose_learning_mechanism as helpers  # noqa: E402

common = official.common
FAMILIES = ["liquidity_full", "liquidity_readout", "shock_size_only",
            "liquidity_unconstrained", "allocation_constant", "allocation_readout"]


def numeric_state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def initial_parameters(model):
    core = model.core if hasattr(model, "core") else model
    with torch.no_grad():
        w = core._log_transient_allocation(core.base_logits).exp()
        r = {"parameters": sum(p.numel() for p in model.parameters()),
             "rates": core.rates().tolist(), "raw_rates": core.raw_rates.tolist(),
             "base_allocation": [float(1-w.sum()), *w.tolist()],
             "base_logits": core.base_logits.tolist(), "beta": core.beta.tolist()}
        if core.gate is None:
            r["gate"] = "absent; constant allocation"
        elif hasattr(core.gate, "coefficients"):
            gamma, delta = core.gate.coefficients()
            r.update({"gate": "compact volume/innovation gate", "effective_gamma": gamma.tolist(),
                      "effective_delta": delta.tolist()})
        else:
            r.update({"gate": "MLP hidden16; final layer zero, seed-specific hidden weights",
                      "gate_last_layer_weight_max_abs": float(core.gate[2].weight.abs().max()),
                      "gate_last_layer_bias_max_abs": float(core.gate[2].bias.abs().max())})
    return r


def changes(model, initial):
    groups = {}
    for name, p in model.named_parameters():
        delta = p.detach().double()-initial[name].double()
        groups[name] = {"n_parameters": p.numel(), "change_l2": float(delta.norm()),
                        "n_changed_exact": int(torch.count_nonzero(delta))}
    state = model.state_dict()
    return {"groups": groups,
            "parameter_change_l2": float(sum(g["change_l2"]**2 for g in groups.values())**.5),
            "all_state_tensors_equal_to_same_seed_initial": all(torch.equal(state[k], v) for k, v in initial.items()),
            "initial_numeric_state_sha256": numeric_state_hash(initial),
            "checkpoint_numeric_state_sha256": numeric_state_hash(state)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/carbon-literature-aligned-runs-20261019d/official_tianjin_v4"))
    parser.add_argument("--data-file", type=Path, default=official.DATA_FILE)
    parser.add_argument("--out", type=Path, default=STUDY/"methods/V4_INITIALIZATION_AND_LEARNING_ZH.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    official.DATA_FILE = args.data_file.resolve()
    plan = official.checked_plan()
    protocol_path = args.run_dir/"PROTOCOL_BEFORE_RUN.json"
    protocol = json.loads(protocol_path.read_text())
    complete_path = args.run_dir/"EXECUTION_COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    assert complete["status"] == "completed"
    assert protocol["source_sha256"] == official.source_hashes()
    assert protocol["statistics_plan_sha256"] == official.PLAN_SHA
    assert protocol["fixed_primary_candidate"] == "allocation_readout"
    assert protocol["prespecified_secondary_candidate"] == "liquidity_full"
    d = official.prepare_official()
    cell = args.run_dir/f"{official.MARKET}_{official.FOLD}"
    manifest_path = cell/"INPUT_MANIFEST.json"
    assert json.loads(manifest_path.read_text()) == d["manifest"]
    masks = d["masks"]
    y_test = d["y"][masks["test"]]
    read_hashes = {str(p.relative_to(args.run_dir)): helpers.digest(p)
                   for p in [protocol_path, complete_path, manifest_path]}
    records, families, initial_predictions = [], {}, {}
    max_validation_gap = max_initial_validation_gap = max_prediction_gap = max_seed_initial_gap = 0.
    for family in FAMILIES:
        directory = cell/family
        choice_path = directory/"CHOICE_BEFORE_TEST.json"
        choice = json.loads(choice_path.read_text())
        summary_path = directory/"SUMMARY.json"
        summary = json.loads(summary_path.read_text())
        predictions_path = directory/"predictions.npz"
        for p in [choice_path, summary_path, predictions_path]:
            read_hashes[str(p.relative_to(args.run_dir))] = helpers.digest(p)
        with np.load(predictions_path) as saved:
            original = saved["predictions"].copy()
            assert np.array_equal(saved["observed"], y_test)
            assert np.array_equal(saved["origins"], d["origins"][masks["test"]])
            assert np.array_equal(saved["horizons"], common.HORIZONS)
            assert np.array_equal(saved["seeds"], common.SEEDS)
        initial_models = {}
        for seed in common.SEEDS:
            model = official.make_official_model(family, seed)
            state = copy.deepcopy(model.state_dict())
            pred = common.predict(model, d["x"])
            initial_models[seed] = {"state": state, "prediction": pred,
                                    "numeric_state_sha256": numeric_state_hash(state)}
        initial_test = initial_models[17]["prediction"][masks["test"]]
        initial_predictions[family] = initial_test
        for seed in [29, 43]:
            gap = float(np.max(np.abs(initial_models[seed]["prediction"]-initial_models[17]["prediction"])))
            max_seed_initial_gap = max(max_seed_initial_gap, gap)
            assert gap == 0
        initial_risk = helpers.risk_components(initial_test*d["scale"][0], y_test)
        final_risk = helpers.risk_components(original, y_test)
        assert abs(final_risk["risk_over_rw"]-summary["mean_seed_mse_over_rw"]) < 1e-12
        family_records = []
        paths = sorted(directory.glob("selection_lr*.json"))+sorted(directory.glob("refit_seed*.json"))
        assert len(paths) == 5
        for receipt_path in paths:
            receipt = json.loads(receipt_path.read_text())
            checkpoint_path = receipt_path.with_suffix(".pt")
            assert helpers.digest(checkpoint_path) == receipt["checkpoint_sha256"]
            for p in [receipt_path, checkpoint_path]:
                read_hashes[str(p.relative_to(args.run_dir))] = helpers.digest(p)
            seed = receipt["seed"]
            initial = initial_models[seed]
            model = official.make_official_model(family, seed)
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            prediction = common.predict(model, d["x"])
            role = "train" if receipt["mode"] == "train" else "final"
            mask = masks[role]
            a = helpers.exact_loss(initial["prediction"][mask], d, mask)
            b = helpers.exact_loss(prediction[mask], d, mask)
            rec = {"family": family, "seed": seed, "mode": receipt["mode"],
                   "learning_rate": receipt["learning_rate"],
                   "is_original_selected_lr": receipt["learning_rate"] == choice["learning_rate"],
                   "epochs_executed": receipt["epochs_executed"], "selected_epoch": receipt["selected_epoch"],
                   "gradient_positive_steps": receipt["gradient_positive_steps"],
                   "max_gradient_norm_before_clip": receipt["max_gradient_norm_before_clip"],
                   "optimization_parameter_change_before_selection_restore": receipt["optimization_parameter_change_l2_before_selection"],
                   "checkpoint_change": changes(model, initial["state"]),
                   "initial_exact_fitting_loss": a, "checkpoint_exact_fitting_loss": b,
                   "fitting_loss_difference": b-a,
                   "checkpoint_sha256": receipt["checkpoint_sha256"],
                   "receipt": str(receipt_path.relative_to(args.run_dir))}
            if role == "train":
                val = helpers.exact_loss(prediction[masks["val"]], d, masks["val"])
                initial_val = helpers.exact_loss(initial["prediction"][masks["val"]], d, masks["val"])
                gap = abs(val-receipt["selected_validation_loss"])
                init_gap = abs(initial_val-receipt["initial_validation_loss"])
                max_validation_gap = max(max_validation_gap, gap)
                max_initial_validation_gap = max(max_initial_validation_gap, init_gap)
                assert max(gap, init_gap) < 1e-12
                history = receipt["history"]
                trained_min = min(h["validation_loss"] for h in history)
                rec.update({"initial_validation_loss_replayed": initial_val,
                            "checkpoint_validation_loss_replayed": val,
                            "validation_replay_gap": gap, "initial_validation_replay_gap": init_gap,
                            "minimum_logged_trained_epoch_validation_loss": trained_min,
                            "minimum_logged_trained_validation_minus_initial": trained_min-initial_val,
                            "logged_first_epoch_minibatch_train_loss": history[0]["train_loss"],
                            "logged_last_epoch_minibatch_train_loss": history[-1]["train_loss"],
                            "logged_last_epoch_validation_loss": history[-1]["validation_loss"],
                            "logged_training_losses_are_not_checkpoint_full_data_risks": True})
            else:
                native = prediction[masks["test"]]*d["scale"][0]
                target = original[common.SEEDS.index(seed)]
                gap = float(np.max(np.abs(native-target)))
                shift = float(np.max(np.abs(prediction-initial["prediction"])))
                max_prediction_gap = max(max_prediction_gap, gap)
                assert gap == 0
                rec.update({"original_prediction_replay_max_abs_gap_native": gap,
                            "forecast_max_abs_shift_from_same_seed_initial_normalized_all_origins": shift})
            if receipt["selected_epoch"] == 0:
                assert rec["checkpoint_change"]["all_state_tensors_equal_to_same_seed_initial"]
            records.append(rec)
            family_records.append(rec)
        selection = [r for r in family_records if r["mode"] == "train"]
        refits = [r for r in family_records if r["mode"] == "final"]
        families[family] = {
            "initial_parameters": initial_parameters(official.make_official_model(family, 17)),
            "initial_numeric_state_sha256_by_seed": {str(s): initial_models[s]["numeric_state_sha256"] for s in common.SEEDS},
            "initial_risk": initial_risk, "final_risk": final_risk,
            "learning_improvement_ratio_points": initial_risk["risk_over_rw"]-final_risk["risk_over_rw"],
            "prior_improvement_from_rw_ratio_points": 1-initial_risk["risk_over_rw"],
            "selected_lr": choice["learning_rate"], "selected_epoch": choice["selected_epoch"],
            "selection_tasks": len(selection), "selection_tasks_with_positive_steps": sum(r["gradient_positive_steps"] > 0 for r in selection),
            "selection_positive_steps": sum(r["gradient_positive_steps"] for r in selection),
            "refit_constructions": len(refits), "refit_tasks_with_positive_steps": sum(r["gradient_positive_steps"] > 0 for r in refits),
            "refit_positive_steps": sum(r["gradient_positive_steps"] for r in refits),
            "all_five_saved_checkpoints_equal_same_seed_initial": all(r["checkpoint_change"]["all_state_tensors_equal_to_same_seed_initial"] for r in family_records),
            "refit_max_forecast_shift_from_initial_normalized": max(r["forecast_max_abs_shift_from_same_seed_initial_normalized_all_origins"] for r in refits),
        }
    assert len(records) == 30
    assert np.array_equal(initial_predictions["allocation_constant"], initial_predictions["allocation_readout"])
    assert np.array_equal(initial_predictions["liquidity_full"], initial_predictions["liquidity_unconstrained"])
    assert official.source_hashes() == protocol["source_sha256"]
    official.checked_plan()
    selection = [r for r in records if r["mode"] == "train"]
    refits = [r for r in records if r["mode"] == "final"]
    report = {
        "status": "PASS_READ_ONLY_REPLAY", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-result mechanism diagnostic of six fixed structure families; not a new statistical contrast or candidate selection.",
        "authorship": "Model implementer and v4 adapter author; not external blind review.",
        "new_fit_count": 0, "new_optimizer_steps": 0, "new_candidate_selections": 0, "new_significance_tests": 0,
        "market": official.MARKET, "fold": official.FOLD, "primary_candidate": "allocation_readout", "secondary_candidate": "liquidity_full",
        "counts": d["manifest"]["counts"], "input_manifest": d["manifest"],
        "aggregation": "Single official Tianjin tail, equal H1/H5/H20 mean of individual-seed MSE/RW; no cross-market macro.",
        "initialization_interpretation": "All three seeds have identical initial prediction functions; v1 MLP hidden weights remain seed-specific, so numerical state comparisons use the same seed.",
        "numeric_state_hash_definition": "SHA256 of sorted tensor names/dtypes/shapes JSON headers and contiguous CPU numpy bytes; distinct from serialized pt byte SHA.",
        "checks": {"checkpoint_count": len(records), "selection_checkpoint_count": len(selection), "refit_checkpoint_count": len(refits),
                   "maximum_selected_validation_loss_replay_gap": max_validation_gap,
                   "maximum_initial_validation_loss_replay_gap": max_initial_validation_gap,
                   "maximum_original_prediction_replay_gap_native": max_prediction_gap,
                   "maximum_initial_prediction_gap_across_seeds": max_seed_initial_gap,
                   "all_saved_state_dicts_equal_same_seed_initial": all(r["checkpoint_change"]["all_state_tensors_equal_to_same_seed_initial"] for r in records),
                   "all_refit_predictions_equal_own_initial": all(r["forecast_max_abs_shift_from_same_seed_initial_normalized_all_origins"] == 0 for r in refits),
                   "readout_and_constant_predictions_equal": bool(np.array_equal(initial_predictions["allocation_constant"], initial_predictions["allocation_readout"])),
                   "full_and_unconstrained_predictions_equal": bool(np.array_equal(initial_predictions["liquidity_full"], initial_predictions["liquidity_unconstrained"]))},
        "original_optimization_counts": {"selection_tasks": len(selection), "selection_tasks_with_positive_steps": sum(r["gradient_positive_steps"] > 0 for r in selection),
                                         "selection_positive_steps": sum(r["gradient_positive_steps"] for r in selection),
                                         "selection_tasks_with_nonzero_pre_restore_parameter_change": sum(r["optimization_parameter_change_before_selection_restore"] > 0 for r in selection),
                                         "refit_constructions": len(refits), "refit_tasks_with_positive_steps": sum(r["gradient_positive_steps"] > 0 for r in refits),
                                         "refit_positive_steps": sum(r["gradient_positive_steps"] for r in refits)},
        "families": families, "checkpoints": records,
        "source_sha256": protocol["source_sha256"], "source_hashes_unchanged": True, "statistics_plan_sha256": official.PLAN_SHA,
        "diagnostic_script_sha256": helpers.digest(__file__), "helper_script_sha256": helpers.digest(Path(helpers.__file__)),
        "source_artifact_hashes_relative_to_run": read_hashes,
        "runtime": {"python_executable": sys.executable, "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"out": str(args.out), "checks": report["checks"], "counts": report["original_optimization_counts"],
                      "risks": {f: {"init": z["initial_risk"]["risk_over_rw"], "final": z["final_risk"]["risk_over_rw"], "learning": z["learning_improvement_ratio_points"]} for f, z in families.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
