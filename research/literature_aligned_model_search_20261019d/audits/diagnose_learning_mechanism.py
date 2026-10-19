"""Read-only diagnostics of completed v1 checkpoint learning and mechanisms.

No fit, optimizer, backward pass, checkpoint rewrite, or performance selection.
Fixed-weight interventions measure forecast effects only; their errors against
development outcomes are deliberately not computed.  Existing checkpoint and
initialization losses are retrospective diagnostics of an exposed experiment.
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
from torch import nn


STUDY = Path(__file__).resolve().parents[1]
SOURCE = STUDY / "source"
sys.path.insert(0, str(SOURCE))
import run_model_search as common  # noqa: E402


FAMILIES = ("allocation_full", "allocation_readout", "allocation_constant")
GROUPS = ("raw_rates", "base_logits", "beta", "gate.0", "gate.2")
INTERVENTIONS = ("gate_off", "beta_off", "gate_energy_off", "gate_magnitude_off")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return float(value)


def moments(a):
    a = np.asarray(a, dtype=float)
    return {
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "minimum": a.min(axis=0).tolist(),
        "q05": np.quantile(a, .05, axis=0).tolist(),
        "median": np.quantile(a, .5, axis=0).tolist(),
        "q95": np.quantile(a, .95, axis=0).tolist(),
        "maximum": a.max(axis=0).tolist(),
    }


def exact_loss(pred_normalized, d, mask):
    native = np.asarray(pred_normalized, dtype=float) * d["scale"][0]
    return number(np.mean((native-d["y"][mask])**2/d["variance"]))


def risk_components(native_pred, y):
    # Works with one forecast or a leading refit seed dimension.
    p = np.asarray(native_pred, dtype=float)
    if p.ndim == 2:
        p = p[None]
    rw = np.mean(y*y, axis=0)
    energy = np.mean(p*p, axis=1)/rw[None]
    alignment = 2*np.mean(p*y[None], axis=1)/rw[None]
    risk = np.mean((p-y[None])**2, axis=1)/rw[None]
    assert np.max(np.abs(risk-(1+energy-alignment))) < 1e-12
    return {
        "risk_over_rw": number(risk.mean()),
        "forecast_energy_penalty": number(energy.mean()),
        "outcome_alignment_credit": number(alignment.mean()),
        "risk_by_seed_horizon": risk.tolist(),
        "energy_by_seed_horizon": energy.tolist(),
        "alignment_by_seed_horizon": alignment.tolist(),
    }


def parameter_record(model, initial_state):
    state = model.state_dict()
    groups = {}
    for group in GROUPS:
        names = [k for k in state if k == group or k.startswith(group+".")]
        if not names:
            continue
        v = torch.cat([state[k].flatten().double() for k in names])
        i = torch.cat([initial_state[k].flatten().double() for k in names])
        delta = v-i
        groups[group] = {
            "n_parameters": v.numel(), "initial_l2": number(i.norm()),
            "checkpoint_l2": number(v.norm()), "change_l2": number(delta.norm()),
            "max_abs_change": number(delta.abs().max()),
            "n_changed_exact": int(torch.count_nonzero(delta)),
        }
    with torch.no_grad():
        rho = model.rates().double().numpy()
        temp = model._log_transient_allocation(model.base_logits).exp().double().numpy()
    a = np.diag(rho)+np.outer(temp, 1-rho)
    return {
        "parameters": model.parameter_count(),
        "rates": rho.tolist(), "base_allocation": [number(1-temp.sum()), *temp.tolist()],
        "beta": state["beta"].tolist(), "groups": groups,
        "checkpoint_change_l2": number(sum(g["change_l2"]**2 for g in groups.values())**.5),
        "constant_allocation_observation_driven_transition_eigenvalues": np.linalg.eigvals(a).real.tolist(),
        "constant_transition_scope": "Exact for readout/constant history; for full this is only its gate-off reference, not the nonlinear trajectory Jacobian.",
        "state_dict": {k: v.tolist() for k, v in state.items()},
    }


def forward_diagnostics(model, x):
    predictions, arrays = [], {}
    with torch.no_grad():
        for j in range(0, len(x), 256):
            p, d = model.forecast_with_diagnostics(torch.from_numpy(x[j:j+256]))
            predictions.append(p.numpy()[:, common.HORIZONS-1])
            for key in ("innovation", "transient_state", "allocation", "proposed_allocation",
                        "gate_adjustment", "readout_multiplier"):
                arrays.setdefault(key, []).append(d[key].numpy())
    z = {k: np.concatenate(v) for k, v in arrays.items()}
    q = z["gate_adjustment"]
    alloc = z["allocation"]
    proposed = z["proposed_allocation"]
    state = z["transient_state"]
    # Historical pairs repeat overlapping observations, as the finite-window
    # model itself reinitializes each origin. Endpoint summaries do not.
    summary = {
        "n_origins": len(x), "n_history_tokens_per_origin": x.shape[1],
        "endpoint_gate_adjustment": moments(q[:, -1]),
        "endpoint_gate_fraction_abs_adjustment_above_1_9": number(np.mean(np.abs(q[:, -1]) > 1.9)),
        "historical_origin_token_gate_adjustment": moments(q.reshape(-1, q.shape[-1])),
        "historical_origin_token_gate_fraction_abs_adjustment_above_1_9": number(np.mean(np.abs(q) > 1.9)),
        "endpoint_actual_allocation": moments(alloc[:, -1]),
        "endpoint_proposed_allocation": moments(proposed[:, -1]),
        "endpoint_temporary_mass": moments(alloc[:, -1, 1:].sum(-1)),
        "endpoint_proposed_temporary_mass": moments(proposed[:, -1, 1:].sum(-1)),
        "endpoint_transient_state": moments(state[:, -1]),
        "endpoint_innovation": moments(z["innovation"][:, -1]),
        "readout_multiplier": moments(z["readout_multiplier"]),
    }
    return np.concatenate(predictions), summary


class FixedInputMask(nn.Module):
    def __init__(self, gate, indices):
        super().__init__()
        self.gate, self.indices = gate, indices

    def forward(self, x):
        y = x.clone()
        y[:, self.indices] = 0
        return self.gate(y)


def intervention_effects(model, x, reference, variance_normalized):
    effects = {}
    baseline_power = number(np.mean(reference.astype(float)**2/variance_normalized))
    for intervention in INTERVENTIONS:
        alternative = copy.deepcopy(model)
        if intervention == "gate_off":
            alternative.gate = None
        elif intervention == "beta_off":
            with torch.no_grad():
                alternative.beta.zero_()
        elif intervention == "gate_energy_off":
            alternative.gate = FixedInputMask(alternative.gate, list(range(3, 3+model.n_exog)))
        else:
            alternative.gate = FixedInputMask(alternative.gate, [-1])
        p = common.predict(alternative, x)
        delta = p.astype(float)-reference
        power = number(np.mean(delta**2/variance_normalized))
        effects[intervention] = {
            "delta_rms_in_training_risk_units": power**.5,
            "delta_rms_over_frozen_forecast_rms": (power/baseline_power)**.5 if baseline_power > 0 else None,
            "delta_rms_by_horizon_normalized_price_units": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
            "mean_delta_by_horizon_normalized_price_units": delta.mean(axis=0).tolist(),
            "maximum_abs_delta_normalized_price_units": number(np.max(np.abs(delta))),
            "fraction_forecast_sign_changed": number(np.mean(p*reference < 0)),
        }
    return effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/carbon-literature-aligned-runs-20261019d/main_v1"))
    parser.add_argument("--out", type=Path, default=STUDY/"methods/LEARNING_AND_MECHANISM_DIAGNOSIS_ZH.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads((args.run_dir/"PROTOCOL_BEFORE_RUN.json").read_text())
    frozen = protocol["source_sha256"]
    assert all(digest(SOURCE/name) == value for name, value in frozen.items())
    checkpoints, cells, files_read = [], [], {}
    maximum_validation_gap = maximum_prediction_gap = 0.0
    for market in common.MARKETS:
        for year in common.YEARS:
            cell = f"{market}_{year}"
            directory = args.run_dir/cell
            d = common.prepare(market, year)
            manifest = json.loads((directory/"INPUT_MANIFEST.json").read_text())
            assert manifest == d["manifest"]
            files_read[str((directory/"INPUT_MANIFEST.json").relative_to(args.run_dir))] = digest(directory/"INPUT_MANIFEST.json")
            mask = d["masks"]["test"]
            x_test, y_test = d["x"][mask], d["y"][mask]
            initial_model = common.make_model("allocation_full", 17)
            initial_test = common.predict(initial_model, x_test)
            initial_risk = risk_components(initial_test*d["scale"][0], y_test)
            cell_report = {"cell": cell, "counts": manifest["counts"],
                           "training_scale": d["scale"].tolist(),
                           "initialization_development_risk": initial_risk,
                           "families": {}}
            for family in FAMILIES:
                directory_f = directory/family
                choice_path = directory_f/"CHOICE_BEFORE_TEST.json"
                choice = json.loads(choice_path.read_text())
                files_read[str(choice_path.relative_to(args.run_dir))] = digest(choice_path)
                with np.load(directory_f/"predictions.npz") as saved:
                    saved_predictions = saved["predictions"].copy()
                    assert np.array_equal(saved["origins"], d["origins"][mask])
                    assert np.array_equal(saved["observed"], y_test)
                    assert np.array_equal(saved["horizons"], common.HORIZONS)
                    assert np.array_equal(saved["seeds"], common.SEEDS)
                files_read[str((directory_f/"predictions.npz").relative_to(args.run_dir))] = digest(directory_f/"predictions.npz")
                family_records = []
                paths = sorted(directory_f.glob("selection_lr*.json"))+sorted(directory_f.glob("refit_seed*.json"))
                assert len(paths) == 5
                for path in paths:
                    receipt = json.loads(path.read_text())
                    checkpoint = path.with_suffix(".pt")
                    assert digest(checkpoint) == receipt["checkpoint_sha256"]
                    files_read[str(path.relative_to(args.run_dir))] = digest(path)
                    files_read[str(checkpoint.relative_to(args.run_dir))] = digest(checkpoint)
                    seed = receipt["seed"]
                    model = common.make_model(family, seed)
                    initial_state = copy.deepcopy(model.state_dict())
                    train_role = "train" if receipt["mode"] == "train" else "final"
                    train_mask = d["masks"][train_role]
                    x_fit = d["x"][train_mask]
                    initial_fit_pred = common.predict(model, x_fit)
                    initial_fit_loss = exact_loss(initial_fit_pred, d, train_mask)
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                    model.eval()
                    parameter_info = parameter_record(model, initial_state)
                    fit_prediction = common.predict(model, x_fit)
                    fit_loss = exact_loss(fit_prediction, d, train_mask)
                    history = receipt["history"]
                    entry = {
                        "cell": cell, "family": family, "file": str(path.relative_to(args.run_dir)),
                        "mode": receipt["mode"], "seed": seed, "learning_rate": receipt["learning_rate"],
                        "selected_epoch": receipt["selected_epoch"], "epochs_executed": receipt["epochs_executed"],
                        "is_original_selected_lr": receipt["learning_rate"] == choice["learning_rate"],
                        "gradient_positive_steps": receipt["gradient_positive_steps"],
                        "max_gradient_norm_before_clip": receipt["max_gradient_norm_before_clip"],
                        "optimization_change_before_best_state_restore": receipt["optimization_parameter_change_l2_before_selection"],
                        "parameters": parameter_info,
                        "initial_exact_fitting_loss": initial_fit_loss,
                        "checkpoint_exact_fitting_loss": fit_loss,
                        "checkpoint_fitting_loss_relative_change_percent": 100*(fit_loss/initial_fit_loss-1),
                        "logged_first_epoch_minibatch_mean_loss": history[0]["train_loss"] if history else None,
                        "logged_last_epoch_minibatch_mean_loss": history[-1]["train_loss"] if history else None,
                        "logged_last_over_first_train_loss": history[-1]["train_loss"]/history[0]["train_loss"] if history else None,
                        "checkpoint_sha256": receipt["checkpoint_sha256"],
                    }
                    if receipt["mode"] == "train":
                        val_mask = d["masks"]["val"]
                        val_prediction = common.predict(model, d["x"][val_mask])
                        val_loss = exact_loss(val_prediction, d, val_mask)
                        gap = abs(val_loss-receipt["selected_validation_loss"])
                        maximum_validation_gap = max(maximum_validation_gap, gap)
                        assert gap < 1e-6
                        entry.update({
                            "initial_validation_loss": receipt["initial_validation_loss"],
                            "checkpoint_validation_loss": val_loss,
                            "validation_replay_absolute_gap": gap,
                            "validation_loss_relative_change_percent": 100*(val_loss/receipt["initial_validation_loss"]-1),
                            "last_executed_validation_loss": history[-1]["validation_loss"],
                        })
                        if entry["is_original_selected_lr"]:
                            _, entry["validation_gate_and_state"] = forward_diagnostics(model, d["x"][val_mask])
                    else:
                        pred_test, diagnostics_test = forward_diagnostics(model, x_test)
                        native = pred_test*d["scale"][0]
                        i = common.SEEDS.index(seed)
                        gap = number(np.max(np.abs(native-saved_predictions[i])))
                        maximum_prediction_gap = max(maximum_prediction_gap, gap)
                        assert gap < 1e-6
                        _, diagnostics_fit = forward_diagnostics(model, x_fit)
                        entry.update({
                            "development_prediction_replay_max_abs_gap_native_units": gap,
                            "development_risk": risk_components(saved_predictions[i], y_test),
                            "fitting_gate_and_state": diagnostics_fit,
                            "development_gate_and_state": diagnostics_test,
                            "development_function_shift_from_initial_rms_in_training_risk_units": number(np.sqrt(np.mean((pred_test.astype(float)-initial_test)**2/(d["variance"]/d["scale"][0]**2)))),
                        })
                        if model.gate is not None:
                            entry["fixed_weight_intervention_forecast_effects"] = intervention_effects(model, x_test, pred_test, d["variance"]/d["scale"][0]**2)
                    if entry["selected_epoch"] == 0:
                        assert parameter_info["checkpoint_change_l2"] == 0
                    checkpoints.append(entry)
                    family_records.append(entry)
                refits = [z for z in family_records if z["mode"] == "final"]
                selected = [z for z in family_records if z["mode"] == "train" and z["is_original_selected_lr"]][0]
                risk = risk_components(saved_predictions, y_test)
                summary = json.loads((directory_f/"SUMMARY.json").read_text())
                assert abs(risk["risk_over_rw"]-summary["mean_seed_mse_over_rw"]) < 1e-12
                cell_report["families"][family] = {
                    "choice_learning_rate": choice["learning_rate"], "choice_epoch": choice["selected_epoch"],
                    "selected_validation_relative_change_percent": selected["validation_loss_relative_change_percent"],
                    "selected_checkpoint_fitting_relative_change_percent": selected["checkpoint_fitting_loss_relative_change_percent"],
                    "refit_fitting_relative_change_percent_mean": number(np.mean([z["checkpoint_fitting_loss_relative_change_percent"] for z in refits])),
                    "refit_changed_parameter_group_counts": {g: sum(z["parameters"]["groups"].get(g, {}).get("change_l2", 0) > 0 for z in refits) for g in GROUPS},
                    "refit_rates_mean": np.mean([z["parameters"]["rates"] for z in refits], axis=0).tolist(),
                    "refit_base_allocation_mean": np.mean([z["parameters"]["base_allocation"] for z in refits], axis=0).tolist(),
                    "refit_beta_mean": np.mean([z["parameters"]["beta"] for z in refits], axis=0).tolist(),
                    "refit_beta_l2_mean": number(np.mean([np.linalg.norm(z["parameters"]["beta"]) for z in refits])),
                    "refit_gate_adjustment_std_endpoint_mean": np.mean([z["development_gate_and_state"]["endpoint_gate_adjustment"]["std"] for z in refits], axis=0).tolist(),
                    "refit_gate_abs_above_1_9_endpoint_mean": number(np.mean([z["development_gate_and_state"]["endpoint_gate_fraction_abs_adjustment_above_1_9"] for z in refits])),
                    "refit_readout_multiplier_mean": np.mean([z["development_gate_and_state"]["readout_multiplier"]["mean"] for z in refits], axis=0).tolist(),
                    "development_risk": risk,
                    "development_risk_change_from_initial_ratio_points": risk["risk_over_rw"]-initial_risk["risk_over_rw"],
                    "intervention_effect_rms_over_forecast_mean": {
                        name: number(np.mean([z["fixed_weight_intervention_forecast_effects"][name]["delta_rms_over_frozen_forecast_rms"] for z in refits]))
                        for name in INTERVENTIONS
                    } if family != "allocation_constant" else None,
                }
            cells.append(cell_report)
            print(f"READ_ONLY_DIAGNOSED {cell}", flush=True)
    overall = {}
    init_mean = number(np.mean([c["initialization_development_risk"]["risk_over_rw"] for c in cells]))
    for family in FAMILIES:
        records = [z for z in checkpoints if z["family"] == family]
        refits = [z for z in records if z["mode"] == "final"]
        selections = [z for z in records if z["mode"] == "train"]
        selected = [z for z in selections if z["is_original_selected_lr"]]
        risks = [c["families"][family]["development_risk"]["risk_over_rw"] for c in cells]
        learning_gains = [c["initialization_development_risk"]["risk_over_rw"]-c["families"][family]["development_risk"]["risk_over_rw"] for c in cells]
        net_gain = sum(learning_gains)
        overall[family] = {
            "mean_seed_development_mse_over_rw_equal_cells_horizons": number(np.mean(risks)),
            "n_cells_better_than_rw": sum(v < 1 for v in risks),
            "n_cells_better_than_initialization": sum(v > 1e-12 for v in learning_gains),
            "n_cells_worse_than_initialization": sum(v < -1e-12 for v in learning_gains),
            "n_cells_equal_initialization": sum(abs(v) <= 1e-12 for v in learning_gains),
            "n_selection_tasks": len(selections),
            "n_selection_tasks_with_positive_gradient_steps": sum(z["gradient_positive_steps"] > 0 for z in selections),
            "n_selection_saved_checkpoints_epoch_zero": sum(z["selected_epoch"] == 0 for z in selections),
            "n_original_selected_cell_checkpoints_epoch_zero": sum(z["selected_epoch"] == 0 for z in selected),
            "n_refit_constructions": len(refits),
            "n_refits_with_positive_gradient_steps": sum(z["gradient_positive_steps"] > 0 for z in refits),
            "n_refit_checkpoint_unchanged": sum(z["parameters"]["checkpoint_change_l2"] == 0 for z in refits),
            "gradient_positive_steps_selection": sum(z["gradient_positive_steps"] for z in selections),
            "gradient_positive_steps_refit": sum(z["gradient_positive_steps"] for z in refits),
            "mean_selected_checkpoint_fitting_loss_relative_change_percent": number(np.mean([z["checkpoint_fitting_loss_relative_change_percent"] for z in selected])),
            "mean_selected_checkpoint_validation_loss_relative_change_percent": number(np.mean([z["validation_loss_relative_change_percent"] for z in selected])),
            "mean_refit_checkpoint_fitting_loss_relative_change_percent": number(np.mean([z["checkpoint_fitting_loss_relative_change_percent"] for z in refits])),
            "mean_risk_ratio_point_improvement_from_initialization": init_mean-number(np.mean(risks)),
            "net_learning_gain_contributions_by_cell": {
                c["cell"]: {"risk_ratio_point_improvement": gain, "fraction_of_net_sum": gain/net_gain if abs(net_gain) > 1e-15 else None}
                for c, gain in zip(cells, learning_gains)
            },
        }
    assert all(digest(SOURCE/name) == value for name, value in frozen.items())
    report = {
        "status": "PASS_READ_ONLY_REPLAY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Retrospective learning and forecast-mechanism diagnostics after v1 completion, using already exposed 2022/2023 development data.",
        "authorship": "The analyst implemented the model and previously repaired a different study solver; this is not external blind review.",
        "new_fit_count": 0, "new_optimizer_step_count": 0, "new_performance_selection_count": 0,
        "fixed_weight_intervention_policy": "Only forecast changes are summarized; intervention outcome errors are not computed, tuned, ranked, or added as model trials.",
        "risk_accounting": "risk/RW = 1 + mean(prediction^2)/mean(outcome^2) - 2 mean(prediction*outcome)/mean(outcome^2), computed within seed/horizon then equally averaged; no seed-ensemble primary score.",
        "loss_accounting": "Exact checkpoint losses use the original full fitting set and fixed training second moments. Logged minibatch losses are separate and are not mistaken for exact epoch-start/end risks.",
        "overlap_accounting": "Historical gate summaries count (forecast origin, within-window token) pairs and repeat overlapping observations; endpoint summaries count origins. Neither creates independent observations.",
        "intervention_names_fixed_before_replay": list(INTERVENTIONS),
        "initialization_risk_equal_cells_horizons": init_mean,
        "initialization_state": {"rates": [.5, .9], "base_allocation": [.9, .05, .05], "beta": [0, 0, 0], "gate_last_layer": "zero", "forecast": "nonzero two-mode reversion, not RW"},
        "source_sha256": frozen, "source_hashes_unchanged": True,
        "diagnostic_script_sha256": digest(__file__),
        "run_protocol_sha256": digest(args.run_dir/"PROTOCOL_BEFORE_RUN.json"),
        "execution_complete_sha256": digest(args.run_dir/"EXECUTION_COMPLETE.json"),
        "runtime": {"python_executable": sys.executable, "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "checks": {"n_checkpoints_loaded_and_hash_matched": len(checkpoints),
                   "maximum_validation_loss_replay_absolute_gap": maximum_validation_gap,
                   "maximum_original_prediction_replay_absolute_gap_native_units": maximum_prediction_gap},
        "overall": overall, "cells": cells, "checkpoints": checkpoints,
        "source_artifact_hashes_relative_to_original_run_dir": files_read,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"out": str(args.out), "status": report["status"], "checks": report["checks"],
                      "initialization": init_mean, "overall": overall}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
