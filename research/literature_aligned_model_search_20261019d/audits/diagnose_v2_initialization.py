"""Frozen-protocol v2 prior-versus-learning diagnostics, without any fitting.

Loads original selection/refit checkpoints, reconstructs each architecture's
own fixed initialization, and replays observed development inputs. Never
rewrites checkpoints or modifies the four frozen executable source files.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
SOURCE = STUDY / "source"
sys.path.insert(0, str(SOURCE))
import diagnose_learning_mechanism as helpers  # noqa: E402
import run_liquidity_extension as extension  # noqa: E402

common = extension.common
FAMILIES = ("liquidity_full", "liquidity_readout", "shock_size_only", "liquidity_unconstrained")


def parameter_record(model, initial):
    state = model.state_dict()
    groups = {}
    for name, parameter in model.named_parameters():
        now = parameter.detach().double()
        before = initial[name].double()
        delta = now-before
        groups[name] = {
            "n_parameters": now.numel(), "initial": before.tolist(), "value": now.tolist(),
            "change_l2": float(delta.norm()), "max_abs_change": float(delta.abs().max()),
            "n_changed_exact": int(torch.count_nonzero(delta)),
        }
    with torch.no_grad():
        gamma, delta = model.gate.coefficients()
        rho = model.rates()
        temp = model._log_transient_allocation(model.base_logits).exp()
    return {
        "n_trainable_parameters": model.parameter_count(), "groups": groups,
        "parameter_change_l2": sum(g["change_l2"]**2 for g in groups.values())**.5,
        "rates": rho.tolist(), "base_allocation": [float(1-temp.sum()), *temp.tolist()],
        "beta": model.beta.detach().tolist(), "effective_gamma": gamma.tolist(), "effective_delta": delta.tolist(),
        "n_negative_gamma": int((gamma < 0).sum()), "n_negative_delta": int((delta < 0).sum()),
        "state_dict_including_reference_buffer": {k: v.tolist() for k,v in state.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/carbon-literature-aligned-runs-20261019d/liquidity_v2"))
    parser.add_argument("--out", type=Path, default=STUDY/"methods/V2_INITIALIZATION_AND_LEARNING_ZH.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol_path = args.run_dir/"PROTOCOL_BEFORE_RUN.json"
    protocol = json.loads(protocol_path.read_text())
    assert "frozen_prior_diagnostic" in protocol
    hashes = protocol["source_sha256"]
    assert all(helpers.digest(SOURCE/name) == value for name,value in hashes.items())
    files_read = {"PROTOCOL_BEFORE_RUN.json": helpers.digest(protocol_path)}
    cells, checkpoints = [], []
    validation_gap = prediction_gap = 0.0
    for market in protocol["markets"]:
        for year in protocol["years"]:
            cell = f"{market}_{year}"
            directory = args.run_dir/cell
            d = extension.prepare_liquidity(market,year)
            manifest_path = directory/"INPUT_MANIFEST.json"
            assert json.loads(manifest_path.read_text()) == d["manifest"]
            files_read[str(manifest_path.relative_to(args.run_dir))] = helpers.digest(manifest_path)
            test_mask = d["masks"]["test"]
            x_test, y_test = d["x"][test_mask], d["y"][test_mask]
            cell_report = {"cell": cell, "counts": d["manifest"]["counts"], "families": {}}
            initial_predictions = {}
            for family in FAMILIES:
                directory_f = directory/family
                choice_path = directory_f/"CHOICE_BEFORE_TEST.json"
                choice = json.loads(choice_path.read_text())
                files_read[str(choice_path.relative_to(args.run_dir))] = helpers.digest(choice_path)
                initial_model = extension.make_liquidity_model(family,17)
                initial_state = copy.deepcopy(initial_model.state_dict())
                # Compact models have deterministic initial parameters. The
                # discarded base MLP must not introduce surviving randomness.
                for seed in [29,43]:
                    alternate = extension.make_liquidity_model(family,seed).state_dict()
                    assert initial_state.keys() == alternate.keys()
                    assert all(torch.equal(initial_state[k],alternate[k]) for k in initial_state)
                initial_test, initial_diagnostics = helpers.forward_diagnostics(initial_model,x_test)
                initial_predictions[family] = initial_test
                initial_risk = helpers.risk_components(initial_test*d["scale"][0],y_test)
                initial_parameter_record = parameter_record(initial_model,initial_state)
                predictions_path = directory_f/"predictions.npz"
                with np.load(predictions_path) as saved:
                    original = saved["predictions"].copy()
                    assert np.array_equal(saved["observed"], y_test)
                    assert np.array_equal(saved["origins"], d["origins"][test_mask])
                    assert np.array_equal(saved["seeds"],common.SEEDS)
                    assert np.array_equal(saved["horizons"],common.HORIZONS)
                files_read[str(predictions_path.relative_to(args.run_dir))] = helpers.digest(predictions_path)
                actual_risk = helpers.risk_components(original,y_test)
                family_records = []
                paths = sorted(directory_f.glob("selection_lr*.json"))+sorted(directory_f.glob("refit_seed*.json"))
                assert len(paths) == 5
                for path in paths:
                    receipt = json.loads(path.read_text())
                    checkpoint = path.with_suffix(".pt")
                    assert helpers.digest(checkpoint) == receipt["checkpoint_sha256"]
                    files_read[str(path.relative_to(args.run_dir))] = helpers.digest(path)
                    files_read[str(checkpoint.relative_to(args.run_dir))] = helpers.digest(checkpoint)
                    model = extension.make_liquidity_model(family,receipt["seed"])
                    role = "train" if receipt["mode"] == "train" else "final"
                    fit_mask = d["masks"][role]
                    fit_x = d["x"][fit_mask]
                    initial_fit_loss = helpers.exact_loss(common.predict(model,fit_x),d,fit_mask)
                    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
                    model.eval()
                    parameters = parameter_record(model,initial_state)
                    fit_loss = helpers.exact_loss(common.predict(model,fit_x),d,fit_mask)
                    record = {
                        "cell": cell,"family": family,"file":str(path.relative_to(args.run_dir)),
                        "mode":receipt["mode"],"seed":receipt["seed"],"learning_rate":receipt["learning_rate"],
                        "selected_epoch":receipt["selected_epoch"],"epochs_executed":receipt["epochs_executed"],
                        "is_original_selected_lr":receipt["learning_rate"] == choice["learning_rate"],
                        "gradient_positive_steps":receipt["gradient_positive_steps"],
                        "optimization_parameter_change_before_selection_restore":receipt["optimization_parameter_change_l2_before_selection"],
                        "initial_exact_fitting_loss":initial_fit_loss,"checkpoint_exact_fitting_loss":fit_loss,
                        "fitting_loss_relative_change_percent":100*(fit_loss/initial_fit_loss-1),
                        "parameters":parameters,"checkpoint_sha256":receipt["checkpoint_sha256"],
                    }
                    if receipt["mode"] == "train":
                        val_mask=d["masks"]["val"]
                        val_loss=helpers.exact_loss(common.predict(model,d["x"][val_mask]),d,val_mask)
                        gap=abs(val_loss-receipt["selected_validation_loss"])
                        validation_gap=max(validation_gap,gap)
                        assert gap < 1e-6
                        record.update({
                            "initial_validation_loss":receipt["initial_validation_loss"],
                            "checkpoint_validation_loss":val_loss,"validation_replay_absolute_gap":gap,
                            "validation_loss_relative_change_percent":100*(val_loss/receipt["initial_validation_loss"]-1),
                        })
                    else:
                        prediction, diagnostics=helpers.forward_diagnostics(model,x_test)
                        native=prediction*d["scale"][0]
                        i=common.SEEDS.index(receipt["seed"])
                        gap=float(np.max(np.abs(native-original[i])))
                        prediction_gap=max(prediction_gap,gap)
                        assert gap < 1e-6
                        risk=helpers.risk_components(original[i],y_test)
                        record.update({
                            "original_prediction_replay_max_abs_gap_native_units":gap,
                            "development_risk":risk,"development_gate_and_state":diagnostics,
                            "development_risk_ratio_point_improvement_from_own_initialization":initial_risk["risk_over_rw"]-risk["risk_over_rw"],
                            "development_forecast_shift_from_own_initialization_rms_in_training_risk_units":float(np.sqrt(np.mean((prediction.astype(float)-initial_test)**2/(d["variance"]/d["scale"][0]**2)))),
                        })
                    if receipt["selected_epoch"] == 0:
                        assert parameters["parameter_change_l2"] == 0
                    checkpoints.append(record)
                    family_records.append(record)
                refits=[r for r in family_records if r["mode"] == "final"]
                selected=[r for r in family_records if r["mode"] == "train" and r["is_original_selected_lr"]][0]
                summary=json.loads((directory_f/"SUMMARY.json").read_text())
                assert abs(summary["mean_seed_mse_over_rw"]-actual_risk["risk_over_rw"]) < 1e-12
                cell_report["families"][family]={
                    "initial_parameters":initial_parameter_record,"initial_gate_and_state":initial_diagnostics,
                    "fixed_initialization_risk":initial_risk,"original_fitted_risk":actual_risk,
                    "initialization_improvement_from_rw_ratio_points":1-initial_risk["risk_over_rw"],
                    "learning_improvement_from_own_initialization_ratio_points":initial_risk["risk_over_rw"]-actual_risk["risk_over_rw"],
                    "learning_improvement_relative_to_own_initialization_percent":100*(1-actual_risk["risk_over_rw"]/initial_risk["risk_over_rw"]),
                    "selected_lr":choice["learning_rate"],"selected_epoch":choice["selected_epoch"],
                    "selected_validation_relative_change_percent":selected["validation_loss_relative_change_percent"],
                    "selected_fitting_relative_change_percent":selected["fitting_loss_relative_change_percent"],
                    "refit_fitting_relative_change_percent_mean":float(np.mean([r["fitting_loss_relative_change_percent"] for r in refits])),
                    "refit_rates_mean":np.mean([r["parameters"]["rates"] for r in refits],axis=0).tolist(),
                    "refit_base_allocation_mean":np.mean([r["parameters"]["base_allocation"] for r in refits],axis=0).tolist(),
                    "refit_beta_mean":np.mean([r["parameters"]["beta"] for r in refits],axis=0).tolist(),
                    "refit_gamma_mean":np.mean([r["parameters"]["effective_gamma"] for r in refits],axis=0).tolist(),
                    "refit_delta_mean":np.mean([r["parameters"]["effective_delta"] for r in refits],axis=0).tolist(),
                    "n_refits_with_any_negative_gamma":sum(r["parameters"]["n_negative_gamma"] > 0 for r in refits),
                    "n_refits_with_any_negative_delta":sum(r["parameters"]["n_negative_delta"] > 0 for r in refits),
                }
            full=initial_predictions["liquidity_full"]
            unconstrained=initial_predictions["liquidity_unconstrained"]
            assert np.array_equal(full,unconstrained)
            readout=initial_predictions["liquidity_readout"]
            cell_report["initial_pairing"]={
                "full_vs_unconstrained_max_abs_forecast_gap_normalized_units":float(np.max(np.abs(full-unconstrained))),
                "full_vs_readout_max_abs_forecast_gap_normalized_units":float(np.max(np.abs(full-readout))),
                "full_vs_readout_rms_forecast_gap_in_training_risk_units":float(np.sqrt(np.mean((full.astype(float)-readout)**2/(d["variance"]/d["scale"][0]**2)))),
            }
            cells.append(cell_report)
            print(f"READ_ONLY_V2_DIAGNOSED {cell}",flush=True)
    overall={}
    for family in FAMILIES:
        initial=np.array([c["families"][family]["fixed_initialization_risk"]["risk_over_rw"] for c in cells])
        actual=np.array([c["families"][family]["original_fitted_risk"]["risk_over_rw"] for c in cells])
        gains=initial-actual
        rows=[r for r in checkpoints if r["family"] == family]
        selections=[r for r in rows if r["mode"] == "train"]
        refits=[r for r in rows if r["mode"] == "final"]
        overall[family]={
            "fixed_initialization_mean_mse_over_rw":float(initial.mean()),
            "original_mean_seed_mse_over_rw":float(actual.mean()),
            "fixed_prior_improvement_from_rw_ratio_points":float(1-initial.mean()),
            "learning_improvement_from_own_initialization_ratio_points":float(gains.mean()),
            "total_improvement_from_rw_ratio_points":float(1-actual.mean()),
            "fraction_total_rw_improvement_from_learning":float(gains.mean()/(1-actual.mean())),
            "n_cells_fitted_better_than_own_initialization":int((gains > 1e-12).sum()),
            "n_cells_fitted_worse_than_own_initialization":int((gains < -1e-12).sum()),
            "n_cells_equal_own_initialization":int((np.abs(gains) <= 1e-12).sum()),
            "n_selection_tasks":len(selections),
            "n_selection_tasks_with_positive_gradient_steps":sum(r["gradient_positive_steps"] > 0 for r in selections),
            "n_refit_constructions":len(refits),
            "n_refits_with_positive_gradient_steps":sum(r["gradient_positive_steps"] > 0 for r in refits),
            "n_original_selected_cells_epoch_zero":sum(c["families"][family]["selected_epoch"] == 0 for c in cells),
            "gradient_positive_steps_selection":sum(r["gradient_positive_steps"] for r in selections),
            "gradient_positive_steps_refit":sum(r["gradient_positive_steps"] for r in refits),
            "n_refits_with_any_negative_gamma":sum(r["parameters"]["n_negative_gamma"] > 0 for r in refits),
            "n_refits_with_any_negative_delta":sum(r["parameters"]["n_negative_delta"] > 0 for r in refits),
            "net_learning_gain_contributions_by_cell":{
                c["cell"]:{"gain_ratio_points":float(g),"fraction_of_net_sum":float(g/gains.sum()) if abs(gains.sum()) > 1e-15 else None}
                for c,g in zip(cells,gains)
            },
        }
    assert all(helpers.digest(SOURCE/name) == value for name,value in hashes.items())
    kappa=.1
    derivative=1-math.exp(-kappa)
    report={
        "status":"PASS_READ_ONLY_REPLAY",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Frozen v2 protocol secondary initialization diagnostic after completed development run; no new fits or selection.",
        "authorship":"Model implementer performed this analysis; not external blind review.",
        "original_protocol_registered_diagnostic":protocol["frozen_prior_diagnostic"],
        "new_fit_count":0,"new_optimizer_step_count":0,"new_performance_selection_count":0,
        "source_hashes_unchanged":True,"source_sha256":hashes,
        "script_sha256":helpers.digest(__file__),
        "helper_script_sha256":helpers.digest(Path(helpers.__file__)),
        "runtime":{"python_executable":sys.executable,"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},
        "checks":{"n_checkpoints_loaded_and_hash_matched":len(checkpoints),
                  "maximum_validation_loss_replay_absolute_gap":validation_gap,
                  "maximum_original_prediction_replay_absolute_gap_native_units":prediction_gap,
                  "all_compact_initial_parameters_equal_across_three_seeds":True,
                  "all_full_unconstrained_initial_predictions_exactly_equal":True},
        "risk_accounting":"Score each refit separately, then equal mean over seed/horizon/market-year. Initialization has one deterministic forecast, verified seed-invariant. Initialization and fitted outputs use identical original-price labels and risk denominators.",
        "identification_limitations":{
            "full_vs_readout":"Identical count and initial coefficients, different placement with nonzero initial gate: initial functions and initial risks differ. Report each arm's own fixed prior before claiming learning-placement effects.",
            "constrained_vs_unconstrained":"Initial functions identical, but coefficient parameterizations have unequal raw-parameter derivatives and optimizer dynamics; performance difference does not identify sign constraints alone.",
            "parameters_vs_function":"Changes in gamma/delta coexist with changes in rho/base/beta; net initialization-to-fit gains are not gate-only causal attribution.",
            "training_loss_vs_development_loss":"Original validation choices are retained, including epoch0 and budget endpoint; no outcome-informed adjustment is made.",
        },
        "initial_coefficient_derivative_comparison":{
            "effective_coefficient":kappa,"positive_raw_initialization":math.log(math.expm1(kappa)),
            "positive_d_effective_d_raw":derivative,"unconstrained_d_effective_d_raw":1.0,
            "unconstrained_over_positive_raw_loss_gradient_ratio_at_same_initial_function":1/derivative,
            "scope":"Analytic chain rule for a nonzero loss gradient with respect to the same effective gamma/delta at initialization. Not a measured 10.5x overall gradient norm, not a 10.5x loss improvement, and AdamW step effects are not inferred by multiplying raw gradients.",
        },
        "overall":overall,"cells":cells,"checkpoints":checkpoints,
        "source_artifact_hashes_relative_to_original_run_dir":files_read,
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"out":str(args.out),"checks":report["checks"],"overall":overall},indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
