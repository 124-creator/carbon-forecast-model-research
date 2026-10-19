"""Stage f: label-mature linear readout adaptation of frozen raw stage-e models.

No nonlinear model is trained here. Both markets and the reused validation
selection are development evidence. A score-producing run requires a plan whose
``matured_readout`` member exactly equals this program's preflight ``contract``.
--preflight checks identities, states, splits and availability, and never calls
a model basis/forecast, ridge solve, optimizer or forecast loss.

The initial price-return RMS and horizon second moments remain fixed. At issue
t, an updated head may use origin tau iff tau >= 64 and tau + 20 <= t. The
current report is observed before issuing a forecast. All three labels mature
together; early H1/H5 labels do not receive separate updates. Head column RMS
is recalculated with the same eligible origin weights used in the ridge loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
STUDY = SOURCE.parent
E_SOURCE = STUDY.parent / "signed_response_search_20261019e/source"
sys.path.insert(0, str(E_SOURCE))
import run_signed_response_search as core
from signed_response_models import DTYPE, SignedResponse

FAMILIES = ("signed_context_response", "static_response", "amplitude_context",
            "signed_shock_only")
MARKETS = ("TIANJIN_OFFICIAL_MEAN", "CEA_OFFICIAL_MEAN")
STRATEGIES = ("frozen", "expanding_equal", "expanding_half_life126", "rolling126",
              "rolling252")
HORIZONS = (1, 5, 20)
FIRST_ORIGIN, MATURITY = 64, 20
BUDGET_PROTOCOL_SHA256 = "f80ec648275a3c79e23d34ad5d668ebf10cf55c7ec5ec1565995b989dc06b771"
BUDGET_COMPLETION_SHA256 = "25ba1089f6cb9875f81a3a46e535f8768c3fe72e67272f3c9a89e41903127e22"
COLUMN_SCALE_FLOOR = 1e-8
NORMAL_EQUATION_TOLERANCE = 1e-9
FROZEN_REPLAY_TOLERANCE = 1e-10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_hashes():
    names = ("signed_response_models.py", "run_signed_response_search.py",
             "run_neural_references.py", "run_budget_sensitivity.py")
    return {"phase_f/run_matured_readout.py": digest(__file__),
            **{"phase_e/"+name: digest(E_SOURCE/name) for name in names}}


def state_arrays(state):
    return {key: np.asarray(value, dtype=np.float64) for key, value in state.items()}


def state_digest(state):
    serial = {key: value.tolist() for key, value in state_arrays(state).items()}
    return hashlib.sha256(json.dumps(serial, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_frozen_model(family, receipt):
    model = SignedResponse(family)
    state = {key: torch.as_tensor(value, dtype=DTYPE)
             for key, value in receipt["state_dict"].items()}
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise AssertionError("Nonfinite frozen state")
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if model.coefficients.numel() != 3 or model.horizons != HORIZONS:
        raise AssertionError("Only original shared three-coefficient readouts are admitted")
    return model


def verify_checkpoint(path, family, receipt):
    """Only our already-trained, hash-bound checkpoint; no forward or fitting."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["family"] != family:
        raise AssertionError("Checkpoint family mismatch")
    left, right = state_arrays(saved["state_dict"]), state_arrays(receipt["state_dict"])
    if left.keys() != right.keys() or any(not np.array_equal(left[k], right[k]) for k in left):
        raise AssertionError("Numeric JSON is not exactly the binary checkpoint state")


def load_inputs(budget_out, datasets_path=None):
    protocol_path, complete_path = budget_out/"PROTOCOL_BEFORE_RUN.json", budget_out/"EXECUTION_COMPLETE.json"
    if digest(protocol_path) != BUDGET_PROTOCOL_SHA256 or digest(complete_path) != BUDGET_COMPLETION_SHA256:
        raise AssertionError("The completed stage-e max320 source run changed")
    protocol, completed = read_json(protocol_path), read_json(complete_path)
    if not (completed["status"] == "completed" and completed["source_hashes_unchanged"]
            and completed["plan_hash_unchanged"]):
        raise AssertionError("Incomplete or unverified source run")
    actual_e = {name: digest(E_SOURCE/name) for name in protocol["source_sha256"]}
    if actual_e != protocol["source_sha256"]:
        raise AssertionError("Frozen stage-e source identity changed")
    core.check_old_sources()
    if tuple(core.HORIZONS) != HORIZONS or core.LOOKBACK != FIRST_ORIGIN:
        raise AssertionError("Original horizon/window constants changed")
    originals = {spec["market"]: spec for spec in protocol["datasets"]}
    supplied = core.read_datasets(datasets_path) if datasets_path else protocol["datasets"]
    if tuple(spec["market"] for spec in supplied) != MARKETS:
        raise AssertionError("Both prescribed markets, in the prescribed order, are required")
    specs, data, models, bindings = [], {}, {}, []
    for supplied_spec in supplied:
        market = supplied_spec["market"]
        if supplied_spec["role"] != "development":
            raise AssertionError("Already-scored markets cannot be promoted to confirmation")
        if supplied_spec["sha256"] != originals[market]["sha256"]:
            raise AssertionError("Dataset replacement is not part of this experiment")
        # Exact canonical bytes, schema, partitions and numerical preparation
        # determine target identity. A later manifest can describe those same
        # bytes more precisely without becoming a different target. Reuse the
        # source-run semantics; only the identical file's location is replaced.
        spec = dict(originals[market])
        spec["file"] = supplied_spec["file"]
        spec["role"] = "development"
        spec["phase_f_supplied_metadata_notes"] = {
            key: supplied_spec[key] for key in ("role", "price_definition", "source_processing",
                                               "source_url", "source_vintage", "license")
            if key in supplied_spec}
        d = core.prepare(spec)
        old_manifest = read_json(budget_out/f"{market}_{core.FOLD}"/"INPUT_MANIFEST.json")
        for key in ("row_boundaries", "counts", "origin_ranges", "training_rms",
                    "train_target_second_moment"):
            if d["manifest"][key] != old_manifest[key]:
                raise AssertionError(f"Original preparation changed: {market}/{key}")
        specs.append(spec)
        data[market] = d
        for family in FAMILIES:
            folder = budget_out/f"{market}_{core.FOLD}"/family
            selection = read_json(folder/"SELECTION.json")
            selected = selection["selected"]
            receipt_name = selected["receipt"]
            if Path(receipt_name).name != receipt_name or not receipt_name.endswith(".json"):
                raise AssertionError("Selection receipt must be a sibling JSON")
            selection_receipt, refit_receipt = read_json(folder/receipt_name), read_json(folder/"REFIT.json")
            for receipt in (selection_receipt, refit_receipt):
                if (receipt["family"] != family or receipt["config"]["alpha"] != selected["alpha"]
                        or receipt["config"]["maximum_epochs"] != 320
                        or receipt["selected_epoch"] != selected["selected_epoch"]):
                    raise AssertionError("Original selection/refit identity mismatch")
            if selection_receipt["validation_loss"] != selected["validation_loss"] or refit_receipt["validation_loss"] is not None:
                raise AssertionError("Selection and final receipt roles were confused")
            binary_name = receipt_name[:-5]+".pt"
            verify_checkpoint(folder/binary_name, family, selection_receipt)
            verify_checkpoint(folder/"refit.pt", family, refit_receipt)
            selection_model = load_frozen_model(family, selection_receipt)
            final_model = load_frozen_model(family, refit_receipt)
            models[market, family] = {"selection": selection_model, "final": final_model,
                                     "selected": selected, "folder": folder}
            file_names = ("SELECTION.json", receipt_name, binary_name, "REFIT.json", "refit.pt", "predictions.npz")
            binding = {"market": market, "family": family, "alpha": selected["alpha"],
                       "original_learning_rate": selected["learning_rate"],
                       "original_selected_epoch": selected["selected_epoch"],
                       "parameter_counts": selection_model.parameter_counts(),
                       "selection_numeric_state_sha256": state_digest(selection_receipt["state_dict"]),
                       "final_numeric_state_sha256": state_digest(refit_receipt["state_dict"]),
                       "validation_state_source": receipt_name, "test_state_source": "REFIT.json",
                       "files": {str(Path(f"{market}_{core.FOLD}")/family/name): digest(folder/name)
                                 for name in file_names}}
            bindings.append(binding)
    return specs, data, models, bindings


def eligible_history(origins, issue, strategy):
    if strategy not in STRATEGIES or strategy == "frozen":
        raise ValueError("An adaptive strategy is required")
    index = np.flatnonzero((origins >= FIRST_ORIGIN) & (origins+MATURITY <= issue))
    if strategy.startswith("rolling"):
        index = index[-int(strategy[len("rolling"):]):]
    if len(index) == 0:
        raise AssertionError("No mature label; no fallback may inspect a future label")
    age = origins[index[-1]]-origins[index]
    weights = np.exp2(-age/126.) if strategy == "expanding_half_life126" else np.ones(len(index))
    weights /= weights.sum()
    if origins[index[-1]]+MATURITY > issue or not np.isfinite(weights).all() or not (weights > 0).all():
        raise AssertionError("Label maturity/weight check failed")
    return index, weights


def solve_weighted_readout(basis, target, variance, alpha, weights):
    """Exact mean weighted ridge in the original normalized output coordinate.

    Objective = sum_tau w_tau mean_h((Phi beta-y)^2 / v_h)
                + alpha ||s beta||^2, with sum(w)=1 and
    s_j^2 = max(floor^2, sum_tau w_tau mean_h(Phi_j^2 / v_h)).
    No unpenalized intercept, jitter, clipping or pseudoinverse is introduced.
    """
    if (basis.ndim != 3 or basis.shape[:2] != target.shape or basis.shape[1:] != (3, 3)
            or variance.shape != (3,) or weights.shape != (len(basis),) or alpha <= 0
            or not np.isfinite(basis).all() or not np.isfinite(target).all()
            or not np.isfinite(variance).all() or not (variance > 0).all()
            or not np.isfinite(weights).all() or not (weights > 0).all()
            or abs(float(weights.sum())-1.) > 1e-12):
        raise AssertionError("Invalid weighted readout inputs")
    a = basis/np.sqrt(variance)[None, :, None]
    b = target/np.sqrt(variance)[None, :]
    column_scale = np.maximum(np.sqrt(np.einsum("n,nhj,nhj->j", weights, a, a)/3.), COLUMN_SCALE_FLOOR)
    a = a/column_scale
    gram = np.einsum("n,nhi,nhj->ij", weights, a, a)/3.
    rhs = np.einsum("n,nhi,nh->i", weights, a, b)/3.
    regularized = gram+alpha*np.eye(3)
    standardized = np.linalg.solve(regularized, rhs)
    residual = float(np.max(np.abs(regularized@standardized-rhs)))
    if residual > NORMAL_EQUATION_TOLERANCE*(1.+float(np.max(np.abs(rhs)))):
        raise FloatingPointError("Weighted ridge normal equation failed")
    beta = standardized/column_scale
    if not np.isfinite(beta).all():
        raise FloatingPointError("Nonfinite adapted readout")
    return beta, column_scale, residual


def contract_for(specs, data, bindings):
    counts = {market: data[market]["manifest"]["counts"] for market in MARKETS}
    n_solves = 4*len(FAMILIES)*sum(c["val"]+c["test"] for c in counts.values())
    return {"experiment": "raw_matured_readout_on_frozen_stage_e_budget320",
            "analysis_role": "development_for_both_markets_after_prior_scores_seen",
            "primary_family": "signed_context_response", "families": list(FAMILIES),
            "markets": list(MARKETS), "strategies_in_tie_order": list(STRATEGIES),
            "frontend": "unchanged original raw return and raw volume-context tokens; no EWT or dense-path combination",
            "source_sha256": source_hashes(), "budget_protocol_sha256": BUDGET_PROTOCOL_SHA256,
            "budget_completion_sha256": BUDGET_COMPLETION_SHA256,
            "supplied_manifest_rule": "same market/canonical SHA and development role required; only file location overrides source-run spec; original semantics retained, later explanatory wording logged separately; schema/splits/scales must numerically match",
            "datasets": [{"market": s["market"], "sha256": s["sha256"], "role": "development",
                          "row_boundaries": data[s["market"]]["manifest"]["row_boundaries"],
                          "counts": counts[s["market"]],
                          "origin_ranges": data[s["market"]]["manifest"]["origin_ranges"],
                          "initial_training_rms": data[s["market"]]["scale"],
                          "initial_training_target_second_moment": data[s["market"]]["variance"].tolist()}
                         for s in specs],
            "checkpoints": bindings, "horizons": list(HORIZONS), "first_history_origin": FIRST_ORIGIN,
            "label_maturity": "tau+20<=issue_t; report at t is observed; all3 labels update together",
            "old_split": "A=floor(3N/5),B=A+floor(N/5); train tau+20<A; val A<=tau and tau+20<B; final tau+20<B; test B<=tau<=N-21",
            "validation_theta": "original validation-selected training checkpoint; never final-refit theta",
            "test_theta": "original REFIT.json, immutable for the whole test trace",
            "validation_reuse": "Original theta/alpha/epoch already selected on this validation; readout strategy selection is additional development reuse, not nested or prospective confirmation",
            "fixed_normalization": "original price-return RMS and initial-training horizon second moments, including at final/test",
            "weighted_head": "normalized mean over origins and3 horizons; recompute weighted column RMS; same original selected alpha; penalize all3 standardized coefficients including drift",
            "column_scale_floor": COLUMN_SCALE_FLOOR,
            "half_life_unit": "126 target-market observation records in eligible origin age; no calendar-day resampling",
            "rolling_unit": "most recent126/252 eligible origins; all eligible if fewer",
            "effective_weight_count_definition": "1/sum(normalized_origin_weights^2); descriptive weighting count, not a count of independent observations; horizon overlap remains",
            "frozen_strategy": "reuse stored selection head for validation and stored final-refit head for test; zero new ridge solves",
            "first_issue_inclusive_maturity": "At issue A/B the updated policies may add origin A-20/B-20; its H20 label equals the observed issue price",
            "selection": "minimum mean over validation origins/horizons of native squared error / fixed initial-training horizon variance; exact ties use declared strategy order",
            "test_execution": "fix strategy before accessing test forecast losses; issue sequential forecasts, update from matured prior labels only; retain every strategy including unfavorable ones",
            "summary": "native MSE,RMSE,MAE,bias,H-specific MSE/RW MSE and equal-H mean MSE/RW; no new p-values or confirmation gate",
            "frozen_prediction_relative_tolerance": FROZEN_REPLAY_TOLERANCE,
            "normal_equation_relative_tolerance": NORMAL_EQUATION_TOLERANCE,
            "planned_nonlinear_fits": 0, "planned_optimizer_steps": 0,
            "planned_ridge_solves": n_solves, "planned_validation_strategy_paths": 40,
            "planned_test_strategy_paths": 40,
            "publication": "aggregate receipts may be exported; per-origin labels, forecasts and head paths stay in private run output"}


def preflight_boundaries(data):
    records = []
    for market in MARKETS:
        d = data[market]
        _, a, b, n = d["manifest"]["row_boundaries"]
        for subset, issue in (("val", a), ("test", b)):
            for strategy in STRATEGIES[1:]:
                index, weights = eligible_history(d["origins"], issue, strategy)
                last = int(d["origins"][index[-1]])
                if last != issue-20 or last+20 != issue:
                    raise AssertionError("First-issue inclusive maturity was lost")
                limit = int(strategy[7:]) if strategy.startswith("rolling") else None
                expected = min(issue-20-FIRST_ORIGIN+1, limit) if limit else issue-20-FIRST_ORIGIN+1
                if len(index) != expected:
                    raise AssertionError("Wrong warm-start or rolling range")
                # An arbitrarily large unavailable future label cannot enter:
                # eligibility is a pure origin-index operation, with <= t.
                extended_origins = np.arange(FIRST_ORIGIN, n+100)
                future_index, future_weights = eligible_history(extended_origins, issue, strategy)
                if (not np.array_equal(extended_origins[future_index], d["origins"][index])
                        or not np.array_equal(future_weights, weights)):
                    raise AssertionError("Future origins changed eligible history")
                records.append({"market": market, "subset": subset, "strategy": strategy,
                                "first_issue": issue, "eligible_count": len(index),
                                "first_eligible_origin": int(d["origins"][index[0]]),
                                "last_eligible_origin": last, "last_observed_label_row": last+20,
                                "effective_weight_count": float(1./np.sum(weights**2)),
                                "unavailable_future_origin_extension_gap": 0})
    return records


def forecast_paths(d, model, issue_mask, alpha, frozen_fit_mask):
    state_before = state_digest(model.state_dict())
    with torch.no_grad():
        basis = model.basis(core.tensor(d["x"])).numpy()
    target = d["y"]/d["scale"]
    variance = d["variance"]/d["scale"]**2
    issue_index = np.flatnonzero(issue_mask)
    issues = d["origins"][issue_index]
    shape = (len(STRATEGIES), len(issues), 3)
    predictions, heads, scales = np.empty(shape), np.empty(shape), np.empty(shape)
    history_count = np.zeros(shape[:2], dtype=np.int64)
    history_first = np.full(shape[:2], -1, dtype=np.int64)
    history_last = np.full(shape[:2], -1, dtype=np.int64)
    effective_count = np.zeros(shape[:2])
    residuals = np.zeros(shape[:2])
    frozen_index = np.flatnonzero(frozen_fit_mask)
    for i, strategy in enumerate(STRATEGIES):
        for j, (row, issue) in enumerate(zip(issue_index, issues)):
            if strategy == "frozen":
                beta = model.coefficients.detach().numpy()
                column_scale = model.head_column_scale.detach().numpy()
                index, weights = frozen_index, np.full(len(frozen_index), 1./len(frozen_index))
                normal_residual = 0.
            else:
                index, weights = eligible_history(d["origins"], int(issue), strategy)
                # This is the only entry point for label-based adaptation.
                # Every selected row's latest label is observed at issue time.
                if np.any(d["origins"][index]+20 > issue):
                    raise AssertionError("Immature labels reached ridge")
                beta, column_scale, normal_residual = solve_weighted_readout(
                    basis[index], target[index], variance, alpha, weights)
            predictions[i, j] = (basis[row]@beta)*d["scale"]
            heads[i, j], scales[i, j] = beta, column_scale
            history_count[i, j] = len(index)
            history_first[i, j] = d["origins"][index[0]]
            history_last[i, j] = d["origins"][index[-1]]
            effective_count[i, j] = 1./np.sum(weights**2)
            residuals[i, j] = normal_residual
    if state_digest(model.state_dict()) != state_before:
        raise AssertionError("A frozen model state was modified")
    return {"origins": issues, "predictions": predictions, "heads": heads, "head_column_scale": scales,
            "eligible_count": history_count, "first_eligible_origin": history_first,
            "last_eligible_origin": history_last, "effective_weight_count": effective_count,
            "normal_equation_residual": residuals,
            "ridge_solves": (len(STRATEGIES)-1)*len(issues)}


def summarize(d, family, strategy, predicted, selected, trace, strategy_index):
    observed = d["y"][d["masks"]["test"]]
    error = predicted-observed
    mse, rw = np.mean(error**2, axis=0), np.mean(observed**2, axis=0)
    if not (rw > 0).all():
        raise AssertionError("A zero RW denominator must not be silently replaced")
    return {"market": d["manifest"]["market"], "role": "development", "family": family,
            "strategy": strategy, "selected_by_validation": strategy == selected,
            "n_origins": len(observed), "horizons": list(HORIZONS),
            "mse_by_horizon": mse.tolist(), "rmse_by_horizon": np.sqrt(mse).tolist(),
            "mae_by_horizon": np.mean(np.abs(error), axis=0).tolist(),
            "bias_by_horizon": np.mean(error, axis=0).tolist(), "rw_mse_by_horizon": rw.tolist(),
            "mse_over_rw_by_horizon": (mse/rw).tolist(), "mean_mse_over_rw": float(np.mean(mse/rw)),
            "fixed_initial_variance_weighted_loss": float(np.mean(error**2/d["variance"])),
            "readout_solves_test": 0 if strategy == "frozen" else len(observed),
            "test_eligible_count_range": [int(trace["eligible_count"][strategy_index].min()),
                                           int(trace["eligible_count"][strategy_index].max())],
            "test_effective_weight_count_range": [float(trace["effective_weight_count"][strategy_index].min()),
                                                  float(trace["effective_weight_count"][strategy_index].max())],
            "test_normal_equation_max_residual": None if strategy == "frozen" else float(trace["normal_equation_residual"][strategy_index].max()),
            "frozen_normal_equation_trace_zeros_are_not_recomputed_residuals": strategy == "frozen",
            "nonlinear_parameters_changed": False, "new_fitted_parameter_dimensions": 0}


def run_job(d, family, source_models, dest):
    dest.mkdir(parents=True, exist_ok=False)
    choice = source_models["selected"]
    val_trace = forecast_paths(d, source_models["selection"], d["masks"]["val"], choice["alpha"], d["masks"]["train"])
    observed_val = d["y"][d["masks"]["val"]]
    val_losses = np.mean((val_trace["predictions"]-observed_val[None])**2/d["variance"][None, None], axis=(1, 2))
    if not np.isfinite(val_losses).all():
        raise FloatingPointError("Nonfinite validation strategy loss")
    original_val_gap = abs(float(val_losses[0])-choice["validation_loss"])
    if original_val_gap > FROZEN_REPLAY_TOLERANCE*(1.+abs(choice["validation_loss"])):
        raise AssertionError("Frozen selection checkpoint did not replay its original validation loss")
    selected_index = min(range(len(STRATEGIES)), key=lambda i: (float(val_losses[i]), i))
    selected = STRATEGIES[selected_index]
    selection = {"family": family, "alpha_unchanged": choice["alpha"], "selected_strategy": selected,
                 "tie_order": list(STRATEGIES), "validation_theta_source": choice["receipt"],
                 "test_theta_source": "REFIT.json", "frozen_original_validation_loss_gap": original_val_gap,
                 "trials": [{"strategy": s, "validation_loss": float(val_losses[i])}
                            for i, s in enumerate(STRATEGIES)],
                 "test_loss_used_for_selection": False, "validation_is_reused_development": True}
    # Persist the strategy choice before forecasting or scoring any test path.
    write_json_new(dest/"SELECTION.json", selection)
    test_trace = forecast_paths(d, source_models["final"], d["masks"]["test"], choice["alpha"], d["masks"]["final"])
    with np.load(source_models["folder"]/"predictions.npz", allow_pickle=False) as old:
        if (not np.array_equal(old["origins"], test_trace["origins"])
                or not np.array_equal(old["horizons"], np.asarray(HORIZONS))
                or not np.array_equal(old["observed"], d["y"][d["masks"]["test"]])):
            raise AssertionError("Original frozen forecast panel does not align")
        old_predictions = old["predictions"][0]
        frozen_gap = float(np.max(np.abs(test_trace["predictions"][0]-old_predictions)))
        if frozen_gap > FROZEN_REPLAY_TOLERANCE*(1.+float(np.max(np.abs(old_predictions)))):
            raise AssertionError("Frozen policy did not replay the original refit predictions")
    arrays = {"strategies": np.asarray(STRATEGIES), "horizons": np.asarray(HORIZONS),
              "selected_strategy_index": np.asarray(selected_index),
              "validation_observed": observed_val, "test_observed": d["y"][d["masks"]["test"]],
              "validation_losses": val_losses, "initial_price_rms": np.asarray(d["scale"]),
              "initial_target_second_moment": d["variance"]}
    for prefix, trace in (("validation", val_trace), ("test", test_trace)):
        arrays.update({prefix+"_"+key: value for key, value in trace.items() if key != "ridge_solves"})
    np.savez_compressed(dest/"paths.npz", **arrays)
    summaries = [summarize(d, family, strategy, test_trace["predictions"][i], selected, test_trace, i)
                 for i, strategy in enumerate(STRATEGIES)]
    write_json_new(dest/"ALL_STRATEGY_SUMMARIES.json", summaries)
    write_json_new(dest/"PRIMARY_SELECTED_SUMMARY.json", summaries[selected_index])
    receipt = {"market": d["manifest"]["market"], "family": family, "status": "completed",
               "selected_strategy": selected, "ridge_solves_validation": val_trace["ridge_solves"],
               "ridge_solves_test": test_trace["ridge_solves"], "nonlinear_fits": 0, "optimizer_steps": 0,
               "validation_state_unchanged": True, "final_state_unchanged": True,
               "frozen_test_prediction_max_gap": frozen_gap,
               "maximum_normal_equation_residual": float(max(val_trace["normal_equation_residual"].max(),
                                                                test_trace["normal_equation_residual"].max())),
               "paths_sha256": digest(dest/"paths.npz"), "all_five_strategies_retained": True}
    write_json_new(dest/"EXECUTION_COMPLETE.json", receipt)
    return summaries, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-out", type=Path, default=Path("/tmp/carbon-signed-response-runs-20261019e/budget320"))
    parser.add_argument("--datasets", type=Path, help="Optional relocated identical canonical files; metadata/hashes must match the source run")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    specs, data, models, bindings = load_inputs(args.budget_out.resolve(), args.datasets)
    contract = contract_for(specs, data, bindings)
    supplied_manifest = ({"file": str(args.datasets.resolve()), "sha256": digest(args.datasets),
                          "notes": {s["market"]: s["phase_f_supplied_metadata_notes"] for s in specs}}
                         if args.datasets else None)
    if args.preflight:
        if args.preflight_report is None or args.out is not None:
            parser.error("--preflight requires a fresh --preflight-report and forbids --out")
        checks = preflight_boundaries(data)
        write_json_new(args.preflight_report, {"status": "PASS_NO_MARKET_FIT_NO_SCORE", "contract": contract,
                       "market_fits": 0, "ridge_solves": 0, "optimizer_steps": 0, "model_forward_calls": 0,
                       "new_forecast_scores": 0, "binary_json_exact_state_pairs": 16,
                       "supplied_dataset_manifest": supplied_manifest,
                       "availability_checks": checks, "runtime": core.runtime(),
                       "scope": "Source/data hashes, exact checkpoint states and index/weight checks only"})
        print("PASS_NO_MARKET_FIT_NO_SCORE", flush=True)
        return
    if args.plan is None or not args.expected_plan_sha or args.out is None:
        parser.error("A run requires --plan --expected-plan-sha --out")
    if digest(args.plan) != args.expected_plan_sha:
        raise AssertionError("Frozen phase-f plan digest changed")
    plan = read_json(args.plan)
    if plan.get("matured_readout") != contract:
        raise AssertionError("Frozen plan matured_readout contract differs from this exact implementation and inputs")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new empty output; never overwrite source runs or receipts")
    repository = STUDY.parents[1].resolve()
    if args.out.resolve() == repository or repository in args.out.resolve().parents:
        raise ValueError("Per-origin output must stay outside the public repository")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_new(args.out/"PROTOCOL_BEFORE_RUN.json", {"created_unix": time.time(), "contract": contract,
                   "plan_sha256": args.expected_plan_sha, "plan_file": str(args.plan.resolve()),
                   "budget_out": str(args.budget_out.resolve()), "datasets": specs,
                   "supplied_dataset_manifest": supplied_manifest, "runtime": core.runtime()})
    summaries, executions = [], []
    for market in MARKETS:
        write_json_new(args.out/f"{market}_{core.FOLD}"/"INPUT_MANIFEST.json", data[market]["manifest"])
        for family in FAMILIES:
            rows, receipt = run_job(data[market], family, models[market, family],
                                    args.out/f"{market}_{core.FOLD}"/family)
            summaries.extend(rows)
            executions.append(receipt)
            print(json.dumps({"completed": market+"/"+family,
                              "selected_strategy": receipt["selected_strategy"],
                              "development_only": True}), flush=True)
    write_json_new(args.out/"ALL_SUMMARIES.json", summaries)
    write_json_new(args.out/"PRIMARY_SELECTED_SUMMARIES.json", [r for r in summaries if r["selected_by_validation"]])
    unchanged_sources = source_hashes() == contract["source_sha256"]
    unchanged_plan = digest(args.plan) == args.expected_plan_sha
    # Rehash without rerunning any preparation, forecast or loss.
    unchanged_data = all(digest(s["file"]) == s["sha256"] for s in specs)
    unchanged_supplied_manifest = (args.datasets is None or digest(args.datasets) == supplied_manifest["sha256"])
    unchanged_checkpoints = all(digest(args.budget_out/relative) == sha
                               for binding in bindings for relative, sha in binding["files"].items())
    unchanged_source_receipts = (digest(args.budget_out/"PROTOCOL_BEFORE_RUN.json") == BUDGET_PROTOCOL_SHA256
                                 and digest(args.budget_out/"EXECUTION_COMPLETE.json") == BUDGET_COMPLETION_SHA256)
    total_solves = sum(e["ridge_solves_validation"]+e["ridge_solves_test"] for e in executions)
    verified = (unchanged_sources and unchanged_plan and unchanged_data and unchanged_supplied_manifest and unchanged_checkpoints
                and unchanged_source_receipts and total_solves == contract["planned_ridge_solves"])
    write_json_new(args.out/"EXECUTION_COMPLETE.json", {"status": "completed" if verified else "IDENTITY_OR_COUNT_FAILURE",
                   "source_hashes_unchanged": unchanged_sources, "plan_hash_unchanged": unchanged_plan,
                   "data_hashes_unchanged": unchanged_data, "checkpoint_hashes_unchanged": unchanged_checkpoints,
                   "supplied_dataset_manifest_unchanged": unchanged_supplied_manifest,
                   "upstream_protocol_and_completion_unchanged": unchanged_source_receipts,
                   "new_nonlinear_fits": 0, "new_optimizer_steps": 0, "ridge_solves": total_solves,
                   "validation_strategy_paths": 40, "test_strategy_paths": 40, "summary_rows": len(summaries),
                   "primary_selected_rows": sum(r["selected_by_validation"] for r in summaries),
                   "development_only": True, "all_five_strategies_retained": True, "jobs": executions})
    if not verified:
        raise AssertionError("Final identity/count verification failed; retained output is not released")


if __name__ == "__main__":
    main()
