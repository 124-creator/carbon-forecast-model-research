"""Independent, aggregate-only statistics for the frozen stage-f experiment.

This program imports no training runner and performs no fitting or reselection
on test losses.  It reconstructs the original targets from hash-bound canonical
CSVs, checks every forecast panel against those targets, recomputes all native
metrics, and implements STATISTICAL_PLAN_BEFORE_RUN.json with NumPy alone.
The online policy used by each family is the saved validation-selected policy;
all five policy point scores are also retained.  Neither source rows nor
per-origin forecasts, errors, bootstrap paths, or readout paths are exported.

Run only after all three stage-f runners have completed.  ``--self-check`` uses
synthetic arrays only, reads no market data or results, and writes no files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_name] = "1"
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parent.parent
PLAN_SHA256 = "9c8d3188b5e6235f90f47e54fdd9192e974d71165926936c272f793405eff9d3"
MARKETS = ("TIANJIN_OFFICIAL_MEAN", "CEA_OFFICIAL_MEAN")
H = np.asarray([1, 5, 20], dtype=np.int64)
FOLD = "fixed602020"
FAMILIES = ("signed_context_response", "static_response", "amplitude_context", "signed_shock_only")
MAIN = FAMILIES[0]
EWT_SETTINGS = ("window_past", "train_joint_test_window", "full_series_joint")
ENHANCEMENTS = EWT_SETTINGS + ("dense_path", "validation_selected_matured_readout")
STRATEGIES = ("frozen", "expanding_equal", "expanding_half_life126", "rolling126", "rolling252")
DATA_IDENTITIES = {
    "TIANJIN_OFFICIAL_MEAN": (1077, 646, 861, "85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a"),
    "CEA_OFFICIAL_MEAN": (1251, 750, 1000, "f9f0e09b7500b2bdb4ec8b16c61d5d5fa2b64de63a24c0cf7bd1d269aa2be75d"),
}
OUTPUT_NAMES = ("ENHANCEMENT_STATISTICS.json", "ENHANCEMENT_NATIVE_METRICS.csv",
                "ENHANCEMENT_PAIRED_INTERVALS.csv", "ENHANCEMENT_SELECTION_DIAGNOSTICS.json",
                "SAME_PRIMARY_ONLINE_STRATEGY_DIAGNOSTIC.json",
                "ENHANCEMENT_STATISTICS_REPLAY.json")


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(left, right, message, *, atol=1e-11, rtol=1e-10):
    left, right = np.asarray(left), np.asarray(right)
    check(left.shape == right.shape, message + ": incompatible shapes")
    gap = float(np.max(np.abs(left-right))) if left.size else 0.
    check(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False),
          message + f": maximum difference {gap}")
    return gap


def tracked(path, ledger, kind):
    path = Path(path).resolve()
    key = str(path)
    value = {"path": key, "kind": kind, "sha256": digest(path), "bytes": path.stat().st_size}
    if key in ledger:
        check(ledger[key]["sha256"] == value["sha256"], "An input changed during analysis: " + key)
    else:
        ledger[key] = value
    return path


def read_tracked(path, ledger, kind):
    return read(tracked(path, ledger, kind))


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def assert_plan(path):
    check(digest(path) == PLAN_SHA256, "The frozen stage-f statistical plan changed")
    plan = read(path)
    boot = plan["bootstrap"]
    check(tuple(plan["markets"]) == MARKETS and plan["horizons"] == H.tolist()
          and plan["primary_family"] == MAIN, "Frozen market/horizon/family definition")
    check(boot["draws"] == 50000 and boot["block_lengths_in_order"] == [20, 40]
          and boot["seeds"] == dict(zip(MARKETS, (2026101903, 2026101904))),
          "Frozen resampling counts, order, or seeds")
    check(boot["comparison_families"]["enhancement_5"]["family_size"] == 5
          and boot["comparison_families"]["mechanism_15"]["family_size"] == 15,
          "Frozen comparison families")
    return plan


def completion_receipts(args, ledger):
    records, plan_hashes = {}, set()
    expected = {
        "ewt": {"new_prediction_sets": 30, "reused_prediction_sets": 12, "new_fit_tasks": 462},
        "dense_path": {"summary_rows": 8, "selection_fits": 128, "final_refits": 8},
        "matured_readout": {"validation_strategy_paths": 40, "test_strategy_paths": 40,
                            "summary_rows": 40, "primary_selected_rows": 8},
    }
    for block in ("ewt", "dense_path", "matured_readout"):
        folder = args.run_root/block
        completed = read_tracked(folder/"EXECUTION_COMPLETE.json", ledger, "completed_run")
        protocol = read_tracked(folder/"PROTOCOL_BEFORE_RUN.json", ledger, "run_protocol")
        check(completed["status"] == "completed", "Incomplete block: " + block)
        check(all(completed.get(k) == v for k, v in expected[block].items()),
              "Incomplete declared tasks: " + block)
        unchanged = [value for key, value in completed.items() if key.endswith("unchanged")]
        check(bool(unchanged) and all(v is True for v in unchanged), "Run identity failure: " + block)
        plan_hashes.add(protocol["plan_sha256"])
        original_plan = args.run_plan or Path(protocol.get("plan", protocol.get("plan_path", protocol.get("plan_file", ""))))
        tracked(original_plan, ledger, "frozen_execution_plan")
        check(digest(original_plan) == protocol["plan_sha256"], "Execution plan digest mismatch")
        check(tuple(d["market"] for d in protocol["datasets"]) == MARKETS
              and all(d["role"] == "development" for d in protocol["datasets"]),
              "All current results must remain exposed development")
        records[block] = completed
    check(len(plan_hashes) == 1, "The three blocks did not share one execution plan")
    # Bind the reused raw forecasts to the already published stage-e export.
    receipt = read_tracked(STUDY.parent/"signed_response_search_20261019e/provenance/RESULT_EXPORT_RECEIPT_WITH_BUDGET.json",
                           ledger, "published_raw_export_receipt")
    for group, folder, families in (("budget320", args.raw_response_root, FAMILIES),
                                     ("core", args.raw_classical_root, ("ridge_price", "rw"))):
        identities = {row["filename"]: row["sha256"] for row in receipt["runs"][group]["private_prediction_files"]}
        for market in MARKETS:
            for family in families:
                relative = Path(f"{market}_{FOLD}")/family/"predictions.npz"
                check(digest(folder/relative) == identities[str(relative)], "Raw prediction identity changed")
    return records, next(iter(plan_hashes))


def canonical_data(spec, args, ledger):
    market = spec["market"]
    n_expected, a_expected, b_expected, expected_sha = DATA_IDENTITIES[market]
    path = Path(spec["file"]) if args.data_root is None else args.data_root/Path(spec["file"]).name
    tracked(path, ledger, "private_canonical_input")
    check(spec["role"] == "development" and spec["sha256"] == expected_sha
          and digest(path) == expected_sha, "Canonical input or exposure role changed")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames == ["date", "price", "volume_tons", "turnover_yuan"], "Canonical schema")
        rows = list(reader)
    dates = np.asarray([row["date"] for row in rows], dtype="datetime64[D]")
    prices = np.asarray([float(row["price"]) for row in rows])
    volume = np.asarray([float(row["volume_tons"]) if row["volume_tons"] else np.nan for row in rows])
    n, a, b = len(prices), len(prices)*3//5, len(prices)*3//5+len(prices)//5
    check((n, a, b) == (n_expected, a_expected, b_expected), "Canonical count and chronological splits")
    check(np.isfinite(prices).all() and (prices > 0).all()
          and (np.diff(dates).astype(int) > 0).all(), "Unmodified increasing official series")
    origins = np.arange(64, n-20)
    y = prices[origins[:, None]+H]-prices[origins, None]
    train = origins+20 < a
    val = (origins >= a) & (origins+20 < b)
    final, test = origins+20 < b, origins >= b
    change = np.concatenate(([0.], np.diff(prices)))
    scale = max(float(np.sqrt(np.mean(change[1:a]**2))), 1e-7)
    variance = np.maximum(np.mean(y[train]**2, axis=0), scale**2*1e-6)
    valid = np.isfinite(volume) & (volume >= 0)
    log_volume = np.where(valid, np.log1p(np.maximum(np.where(valid, volume, 0.), 0.)), 0.)
    surprise, usable = np.zeros(n), np.zeros(n)
    for t in range(n):
        history = log_volume[max(0, t-60):t][valid[max(0, t-60):t]]
        if valid[t] and len(history) >= 10:
            surprise[t] = np.clip((log_volume[t]-np.median(history))/max(float(np.std(history)), .25), -5, 5)
            usable[t] = 1.
    z = np.column_stack((np.log1p(np.abs(change[origins[test]]/scale)),
                         surprise[origins[test]]*usable[origins[test]]))
    return {"market": market, "y": y[test], "origins": origins[test], "val_y": y[val],
            "val_origins": origins[val], "variance": variance, "scale": scale, "z_test": z,
            "metadata": {"market": market, "role": "development", "n_source_records": n,
                         "n_test_origins": int(test.sum()), "n_validation_origins": int(val.sum()),
                         "n_train_origins": int(train.sum()), "n_final_train_origins": int(final.sum()),
                         "row_boundaries": [0, a, b, n], "canonical_sha256": expected_sha,
                         "test_origin_range": [int(origins[test][0]), int(origins[test][-1])],
                         "test_origin_date_range": [str(dates[origins[test][0]]), str(dates[origins[test][-1]])],
                         "last_test_label_date": str(dates[origins[test][-1]+20]),
                         "horizons_are_subsequent_observed_records": True}}


def load_panel(folder, d, ledger):
    path = tracked(folder/"predictions.npz", ledger, "private_forecast_panel")
    with np.load(path, allow_pickle=False) as saved:
        prediction = saved["predictions"].copy()
        check(prediction.shape == (1, len(d["origins"]), 3), "One deterministic forecast panel per model")
        check(np.array_equal(saved["origins"], d["origins"]) and np.array_equal(saved["horizons"], H),
              "Every model must use exactly the same origin/horizon order")
        check(np.array_equal(saved["observed"], d["y"]), "Unfiltered canonical targets do not align exactly")
        check(np.asarray(saved["deterministic"]).shape == (1,) and bool(saved["deterministic"][0]),
              "Deterministic model disclosure")
        check(np.isfinite(prediction).all(), "Nonfinite forecast")
    return prediction[0]


def metrics(prediction, y):
    error = prediction-y
    squared = error**2
    mse, rw = np.mean(squared, axis=0), np.mean(y*y, axis=0)
    check(np.isfinite(squared).all() and (rw > 0).all(), "Finite loss and strictly positive native RW denominator")
    return {"Q": float(np.mean(mse/rw)), "mse_by_horizon": mse.tolist(),
            "rmse_by_horizon": np.sqrt(mse).tolist(), "mae_by_horizon": np.mean(np.abs(error), axis=0).tolist(),
            "bias_by_horizon": np.mean(error, axis=0).tolist(), "rw_mse_by_horizon": rw.tolist(),
            "mse_over_rw_by_horizon": (mse/rw).tolist()}, squared


def verify_summary(computed, saved):
    gap = close(computed["Q"], saved["mean_mse_over_rw"], "Runner's aggregate Q")
    for name in ("mse_by_horizon", "rmse_by_horizon", "mae_by_horizon", "rw_mse_by_horizon", "mse_over_rw_by_horizon"):
        if name in saved:
            gap = max(gap, close(computed[name], saved[name], "Runner's aggregate " + name))
    return gap


def rates(raw):
    sigmoid = np.exp(-np.logaddexp(0., -np.asarray(raw)))
    first = .98*sigmoid[..., 0]
    second = first+(.98-first)*sigmoid[..., 1]
    return np.stack((first, second), axis=-1)


def array_summary(values):
    values = np.asarray(values)
    return {"minimum": values.min(axis=0).tolist(), "mean": values.mean(axis=0).tolist(),
            "maximum": values.max(axis=0).tolist()}


def response_diagnostics(folder, family, d, ledger):
    choice = read_tracked(folder/"SELECTION.json", ledger, "validation_selection")
    refit = read_tracked(folder/"REFIT.json", ledger, "self_trained_numeric_refit")
    selected, trials = choice["selected"], choice["trials"]
    candidate = min(trials, key=lambda row: (row["validation_loss"], row["alpha"], row["learning_rate"], row["selected_epoch"]))
    check(candidate == selected and len(trials) == 16, "Complete validation-only alpha/LR choice")
    check(refit["family"] == family and refit["selected_epoch"] == selected["selected_epoch"]
          and refit["config"]["alpha"] == selected["alpha"]
          and refit["config"]["learning_rate"] == selected["learning_rate"]
          and refit["config"]["maximum_epochs"] == 320 and refit["validation_loss"] is None,
          "Fixed-budget fresh refit's selected hyperparameters")
    state = refit["state_dict"]
    rho = rates(state["rate_raw"])
    z = d["z_test"][:, :1] if family == "signed_shock_only" else d["z_test"]
    gate = np.asarray(state["gate"])
    q = 2*np.tanh(z@gate.T)
    if family in (MAIN, "signed_shock_only"):
        future = rates(np.asarray(state["rate_raw"])[None]+q)
    else:
        future = np.broadcast_to(rho, (len(d["y"]), 2))
    return {"selected_alpha": selected["alpha"], "selected_learning_rate": selected["learning_rate"],
            "selected_epoch": selected["selected_epoch"], "maximum_epochs": 320,
            "selected_at_budget_cap": selected["selected_epoch"] == 320,
            "validation_loss": selected["validation_loss"], "selection_trials": 16,
            "parameter_counts": refit["parameter_counts"],
            "historical_rates": rho.tolist(), "gate_weights": gate.tolist(),
            "test_gate_offset_summary": array_summary(q),
            "test_future_rates_summary": array_summary(future),
            "test_amplitude_multiplier_summary": array_summary(np.exp(q)) if family == "amplitude_context" else None,
            "signed_readout_coefficients": state["coefficients"],
            "head_column_scale": state["head_column_scale"],
            "nonlinear_parameter_changes": refit.get("selected_nonlinear_parameter_max_change"),
            "final_refit_positive_gradient_steps": refit["positive_gradient_steps"],
            "selection_positive_gradient_steps": sum(row["positive_gradient_steps"] for row in trials)}


def ridge_diagnostics(folder, ledger):
    choice = read_tracked(folder/"SELECTION.json", ledger, "ridge_validation_selection")
    selected = min(choice["trials"], key=lambda row: (row["validation_loss"], row["alpha"]))
    check(choice["selected"] == selected and len(choice["trials"]) == 8, "Ridge validation-only alpha choice")
    features = choice["n_features"]
    return {"selected_alpha": selected["alpha"], "validation_loss": selected["validation_loss"],
            "selection_trials": 8, "n_features": features,
            "total_fitted_scalars": (features+1)*3, "selected_epoch": None,
            "selected_learning_rate": None, "historical_rates": None, "gate_weights": None}


def information_scope(setting):
    if setting == "full_series_joint":
        return "retrospective_full_sequence_analysis_features_include_future_test_prices"
    if setting == "train_joint_test_window":
        return "joint_available_training_prefix; validation_and_test_use_origin_past_window; historical_snapshot"
    if setting == "dense_path":
        return "past_only_inference; H1_to_H20_future_path_is_matured_training_supervision; historical_snapshot"
    if setting == "validation_selected_matured_readout" or setting.startswith("matured_readout_"):
        return "past_inputs_and_tau_plus_20_le_issue_matured_labels; current_report_available; historical_snapshot"
    return "origin_past_inputs_available_from_current_historical_snapshot"


def collect_market(d, args, ledger):
    market = d["market"]
    rows, losses, diagnostics = [], [], []
    max_summary_gap, online_checks = 0., []
    raw_predictions, raw_diagnostics = {}, {}

    def add(setting, family, predicted, *, saved, diag, strategy=None, selected=None, alias_of=None):
        nonlocal max_summary_gap
        calculated, squared = metrics(predicted, d["y"])
        max_summary_gap = max(max_summary_gap, verify_summary(calculated, saved))
        identifier = setting+"/"+family
        record = {"id": identifier, "market": market, "role": "development", "setting": setting,
                  "family": family, "information_scope": information_scope(setting),
                  "strategy": strategy, "selected_by_validation": selected, "alias_of": alias_of,
                  "n_origins": len(d["y"]), "horizons": H.tolist(), **calculated}
        rows.append(record)
        losses.append(squared)
        diagnostics.append({"id": identifier, "market": market, "role": "development", **diag})

    for family in FAMILIES+("ridge_price", "rw"):
        root = args.raw_response_root if family in FAMILIES else args.raw_classical_root
        folder = root/f"{market}_{FOLD}"/family
        predicted = load_panel(folder, d, ledger)
        summary = read_tracked(folder/"SUMMARY.json", ledger, "raw_reused_summary")
        if family in FAMILIES:
            diag = response_diagnostics(folder, family, d, ledger)
            raw_diagnostics[family] = diag
        elif family == "ridge_price":
            diag = ridge_diagnostics(folder, ledger)
        else:
            check(np.array_equal(predicted, np.zeros_like(predicted)), "RW prediction must be the zero cumulative change")
            diag = {"total_fitted_scalars": 0, "selected_epoch": None}
        raw_predictions[family] = predicted
        add("raw320", family, predicted, saved=summary, diag={"reused_stage_e_result": True, **diag})

    for setting in EWT_SETTINGS:
        for family in FAMILIES+("ridge_price",):
            folder = args.run_root/"ewt"/f"{market}_{FOLD}"/setting/family
            predicted = load_panel(folder, d, ledger)
            summary = read_tracked(folder/"SUMMARY.json", ledger, "new_ewt_summary")
            diag = (response_diagnostics(folder, family, d, ledger) if family in FAMILIES
                    else ridge_diagnostics(folder, ledger))
            add(setting, family, predicted, saved=summary, diag=diag)

    for family in FAMILIES:
        folder = args.run_root/"dense_path"/f"{market}_{FOLD}"/family
        predicted = load_panel(folder, d, ledger)
        summary = read_tracked(folder/"SUMMARY.json", ledger, "dense_path_summary")
        add("dense_path", family, predicted, saved=summary,
            diag={"training_horizons": list(range(1, 21)), **response_diagnostics(folder, family, d, ledger)})

    # Validate policy choices independently from the private validation paths.
    # All five test paths are retained, but none of their losses selects a policy.
    for family in FAMILIES:
        folder = args.run_root/"matured_readout"/f"{market}_{FOLD}"/family
        selection = read_tracked(folder/"SELECTION.json", ledger, "online_validation_selection")
        saved_summaries = read_tracked(folder/"ALL_STRATEGY_SUMMARIES.json", ledger, "online_all_strategy_summaries")
        selected_summary = read_tracked(folder/"PRIMARY_SELECTED_SUMMARY.json", ledger, "online_primary_summary")
        completion = read_tracked(folder/"EXECUTION_COMPLETE.json", ledger, "online_family_completion")
        path = tracked(folder/"paths.npz", ledger, "private_online_paths")
        check(digest(path) == completion["paths_sha256"] and completion["status"] == "completed",
              "Online path digest/completion")
        with np.load(path, allow_pickle=False) as saved:
            check(tuple(saved["strategies"].tolist()) == STRATEGIES and np.array_equal(saved["horizons"], H),
                  "All declared online strategies in their fixed tie order")
            check(np.array_equal(saved["validation_origins"], d["val_origins"])
                  and np.array_equal(saved["test_origins"], d["origins"])
                  and np.array_equal(saved["validation_observed"], d["val_y"])
                  and np.array_equal(saved["test_observed"], d["y"]), "Online data panel alignment")
            close(saved["initial_price_rms"], d["scale"], "Unchanged initial price scale")
            close(saved["initial_target_second_moment"], d["variance"], "Unchanged initial target weights")
            val_prediction = saved["validation_predictions"]
            check(val_prediction.shape == (5, len(d["val_y"]), 3) and np.isfinite(val_prediction).all(),
                  "Finite validation-only online paths")
            val_losses = np.mean((val_prediction-d["val_y"][None])**2/d["variance"][None, None], axis=(1, 2))
            close(val_losses, saved["validation_losses"], "Independent online validation losses")
            check(selection["test_loss_used_for_selection"] is False
                  and selection["tie_order"] == list(STRATEGIES)
                  and tuple(row["strategy"] for row in selection["trials"]) == STRATEGIES,
                  "Online strategy selected before test losses")
            close(val_losses, [row["validation_loss"] for row in selection["trials"]], "Saved online validation trial losses")
            selected_index = min(range(5), key=lambda index: (float(val_losses[index]), index))
            check(int(saved["selected_strategy_index"]) == selected_index
                  and selection["selected_strategy"] == STRATEGIES[selected_index], "Validation-only policy replay")
            prediction = saved["test_predictions"].copy()
            check(prediction.shape == (5, len(d["y"]), 3) and np.isfinite(prediction).all(), "All finite test policy paths")
            frozen_gap = close(prediction[0], raw_predictions[family], "Frozen online policy replays raw320")
            check(tuple(row["strategy"] for row in saved_summaries) == STRATEGIES,
                  "No unfavorable online policy was dropped")
            policies = []
            for index, strategy in enumerate(STRATEGIES):
                # The frozen trajectory is a repeated stored head.  Its zeros
                # in the residual trace are not recomputed normal residuals.
                for prefix, issues in (("validation", d["val_origins"]), ("test", d["origins"])):
                    latest = saved[prefix+"_last_eligible_origin"][index]
                    check(latest.shape == issues.shape and np.all(latest+20 <= issues),
                          "An immature label entered the stored readout path")
                    if strategy != "frozen":
                        check(np.array_equal(latest+20, issues), "Updated policy must include the current report's mature H20 label")
                    if strategy.startswith("rolling"):
                        check(np.all(saved[prefix+"_eligible_count"][index] <= int(strategy[7:])),
                              "Rolling policy window exceeded the declared origin count")
                diag = {**raw_diagnostics[family], "frozen_nonlinear_state": True,
                        "strategy": strategy, "selected_by_validation": index == selected_index,
                        "strategy_validation_loss": float(val_losses[index]),
                        "test_readout_coefficient_summary": array_summary(saved["test_heads"][index]),
                        "test_head_column_scale_summary": array_summary(saved["test_head_column_scale"][index]),
                        "test_eligible_count_range": [int(saved["test_eligible_count"][index].min()), int(saved["test_eligible_count"][index].max())],
                        "test_effective_weight_count_range": [float(saved["test_effective_weight_count"][index].min()), float(saved["test_effective_weight_count"][index].max())],
                        "stored_frozen_head_coefficients_are_not_the_adaptive_path": strategy != "frozen"}
                policies.append(diag)
                add("matured_readout_"+strategy, family, prediction[index], saved=saved_summaries[index],
                    diag=diag, strategy=strategy, selected=index == selected_index,
                    alias_of="raw320/"+family if strategy == "frozen" else None)
            check(selected_summary == saved_summaries[selected_index], "Selected online summary identity")
            add("validation_selected_matured_readout", family, prediction[selected_index], saved=selected_summary,
                diag=policies[selected_index], strategy=STRATEGIES[selected_index], selected=True,
                alias_of="matured_readout_"+STRATEGIES[selected_index]+"/"+family)
        online_checks.append({"family": family, "selected_strategy": STRATEGIES[selected_index],
                              "validation_loss": float(val_losses[selected_index]),
                              "all_five_policies_retained": True, "frozen_prediction_max_gap": frozen_gap,
                              "immature_label_violations": 0})
    check(len(rows) == 49 and len({row["id"] for row in rows}) == 49, "All prespecified model/policy aggregates retained")
    return rows, np.stack(losses), diagnostics, {"maximum_summary_metric_replay_gap": max_summary_gap,
                                               "online_checks": online_checks}


def comparisons(plan):
    records = []
    for setting in ENHANCEMENTS:
        records.append({"comparison_id": "enhancement/"+setting, "comparison_family": "enhancement_5",
                        "main_id": setting+"/"+MAIN, "reference_id": "raw320/"+MAIN})
    for setting in ENHANCEMENTS:
        for comparator in ("static_response", "amplitude_context", "signed_shock_only"):
            records.append({"comparison_id": "mechanism/"+setting+"/"+comparator,
                            "comparison_family": "mechanism_15", "main_id": setting+"/"+MAIN,
                            "reference_id": setting+"/"+comparator})
    for setting in EWT_SETTINGS:
        records.append({"comparison_id": "same_ewt_ridge/"+setting, "comparison_family": "secondary_pointwise_only",
                        "main_id": setting+"/"+MAIN, "reference_id": setting+"/ridge_price"})
    for setting in ENHANCEMENTS:
        records.append({"comparison_id": "rw/"+setting, "comparison_family": "secondary_pointwise_only",
                        "main_id": setting+"/"+MAIN, "reference_id": "raw320/rw"})
    check(len(records) == 28 and len({row["comparison_id"] for row in records}) == 28, "Fixed 28 comparisons")
    families = plan["bootstrap"]["comparison_families"]
    check(sum(row["comparison_family"] == "enhancement_5" for row in records) == families["enhancement_5"]["family_size"]
          and sum(row["comparison_family"] == "mechanism_15" for row in records) == families["mechanism_15"]["family_size"],
          "Declared multiplicity family sizes")
    return records


def bootstrap_q(losses, rw_index, rng, draws, block_length, batch_size):
    """One common origin sample for all model/horizon losses in each draw.

    Integer occurrence counts implement the same resampled means as direct
    gathering, while keeping memory bounded. RNG consumption is row-major and
    independent of batch size. The RW's three denominators are redrawn too.
    """
    n_models, n, n_horizons = losses.shape
    check(n_horizons == 3 and np.isfinite(losses).all(), "Three finite horizon losses")
    flat_losses = np.ascontiguousarray(losses.transpose(1, 0, 2).reshape(n, -1))
    sample_count = math.ceil(n/block_length)
    output = np.full((draws, n_models), np.nan)
    invalid, replay_gap = 0, 0.
    index_hasher = hashlib.sha256()
    before = hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()
    for offset in range(0, draws, batch_size):
        batch = min(batch_size, draws-offset)
        starts = rng.integers(0, n, size=(batch, sample_count), dtype=np.int64)
        indices = ((starts[:, :, None]+np.arange(block_length)[None, None, :]) % n).reshape(batch, -1)[:, :n]
        index_hasher.update(np.ascontiguousarray(indices, dtype="<i8").tobytes())
        counts = np.bincount((indices+np.arange(batch)[:, None]*n).ravel(), minlength=batch*n).reshape(batch, n)
        means = (counts.astype(np.float64)@flat_losses/n).reshape(batch, n_models, 3)
        denominators = means[:, rw_index]
        check(np.isfinite(means).all() and (denominators >= 0).all(), "Finite bootstrap losses and nonnegative RW loss")
        valid = (denominators > 0).all(axis=1)
        invalid += int((~valid).sum())
        output[offset+np.flatnonzero(valid)] = np.mean(means[valid]/denominators[valid, None, :], axis=2)
        # A separate summation route checks pairing, horizon axes, truncation,
        # weighting and RW renormalization on the first four actual draws.
        if offset == 0:
            for j in range(min(batch, 4)):
                direct_mean = losses[:, indices[j], :].mean(axis=1)
                direct_denominator = direct_mean[rw_index]
                check(bool((direct_denominator > 0).all()) == bool(valid[j]), "Invalid-draw handling disagrees with direct gathering")
                if valid[j]:
                    direct_q = np.mean(direct_mean/direct_denominator[None], axis=1)
                    replay_gap = max(replay_gap, close(output[j], direct_q, "Direct gathered bootstrap-Q replay", atol=5e-12, rtol=5e-12))
    valid_draws = np.isfinite(output).all(axis=1)
    check(int((~valid_draws).sum()) == invalid and int(valid_draws.sum()) > 0,
          "One common valid/invalid mask and at least one valid draw")
    after = hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()
    return output[valid_draws], {"draws_requested": draws, "valid_draws": int(valid_draws.sum()),
                               "invalid_zero_rw_draws": invalid, "block_length": block_length,
                               "number_of_blocks_per_draw": sample_count, "draw_length": n,
                               "sample_indices_sha256_int64_little_endian": index_hasher.hexdigest(),
                               "rng_state_before_sha256": before, "rng_state_after_sha256": after,
                               "direct_gather_first_four_draw_Q_max_gap": replay_gap,
                               "rw_denominators_recomputed_per_draw_and_horizon": True}


def paired_statistics(rows, losses, market, plan, batch_size):
    index = {row["id"]: i for i, row in enumerate(rows)}
    points = np.asarray([row["Q"] for row in rows])
    draws = plan["bootstrap"]["draws"]
    rng = np.random.default_rng(plan["bootstrap"]["seeds"][market])
    definitions = comparisons(plan)
    output, receipts = [], []
    for block in plan["bootstrap"]["block_lengths_in_order"]:
        samples, receipt = bootstrap_q(losses, index["raw320/rw"], rng, draws, block, batch_size)
        receipts.append(receipt)
        for item in definitions:
            main, reference = index[item["main_id"]], index[item["reference_id"]]
            effects = 100*(samples[:, reference]-samples[:, main])
            point = float(100*(points[reference]-points[main]))
            pointwise = np.quantile(effects, plan["bootstrap"]["pointwise_interval"], method="linear").tolist()
            family = item["comparison_family"]
            if family in plan["bootstrap"]["comparison_families"]:
                definition = plan["bootstrap"]["comparison_families"][family]
                quantiles = definition["bonferroni_quantiles"]
                family_interval = np.quantile(effects, quantiles, method="linear").tolist()
                family_size = definition["family_size"]
            else:
                quantiles, family_interval, family_size = None, None, None
            reference_q = float(points[reference])
            output.append({"market": market, "role": "development", "block_length": block, **item,
                           "information_scope": rows[main]["information_scope"],
                           "main_Q": float(points[main]), "reference_Q": reference_q,
                           "effect_rw_percentage_points": point,
                           "relative_Q_improvement_percent": float(100*(1-points[main]/reference_q)) if reference_q > 0 else None,
                           "relative_Q_improvement_undefined_if_zero_reference": reference_q == 0,
                           "pointwise_95_interval_rw_percentage_points": pointwise,
                           "familywise_95_interval_rw_percentage_points": family_interval,
                           "bonferroni_family_size": family_size, "bonferroni_quantiles": quantiles,
                           "main_better_in_entire_pointwise_interval": pointwise[0] > 0,
                           "main_worse_in_entire_pointwise_interval": pointwise[1] < 0,
                           "main_better_in_entire_family_interval": None if family_interval is None else family_interval[0] > 0,
                           "main_worse_in_entire_family_interval": None if family_interval is None else family_interval[1] < 0,
                           "effect_by_horizon_rw_percentage_points": (100*(np.asarray(rows[reference]["mse_over_rw_by_horizon"])-np.asarray(rows[main]["mse_over_rw_by_horizon"]))).tolist(),
                           "valid_draws": receipt["valid_draws"], "invalid_zero_rw_draws": receipt["invalid_zero_rw_draws"]})
    check(receipts[0]["rng_state_after_sha256"] == receipts[1]["rng_state_before_sha256"],
          "Block40 must continue the same market's RNG after block20")
    return output, receipts


def self_check():
    # Check weighted-count aggregation against direct resampling, plus RNG and
    # output invariance to batching. A synthetic zero-loss RW must be rejected.
    generator = np.random.default_rng(716)
    losses = generator.uniform(.1, 5., size=(7, 43, 3))
    first, receipt_first = bootstrap_q(losses, 0, np.random.default_rng(119), 137, 20, 17)
    second, receipt_second = bootstrap_q(losses, 0, np.random.default_rng(119), 137, 20, 64)
    batching_gap = close(first, second, "Synthetic batch-size invariance", atol=5e-12, rtol=5e-12)
    check(receipt_first["sample_indices_sha256_int64_little_endian"] == receipt_second["sample_indices_sha256_int64_little_endian"]
          and receipt_first["rng_state_after_sha256"] == receipt_second["rng_state_after_sha256"], "Batch size changed the random stream")
    zeros = losses.copy()
    zeros[0] = 0.
    rejected = False
    try:
        bootstrap_q(zeros, 0, np.random.default_rng(1), 7, 40, 4)
    except AssertionError as error:
        rejected = "at least one valid draw" in str(error)
    check(rejected, "An all-zero RW denominator must not be replaced")
    # A partially zero RW produces excluded draws, while retaining valid ones.
    partial = np.ones((2, 5, 3))
    partial[0, :4] = 0.
    _, partial_receipt = bootstrap_q(partial, 0, np.random.default_rng(3), 200, 1, 31)
    check(0 < partial_receipt["invalid_zero_rw_draws"] < 200, "Partial-zero draws were not explicitly excluded")
    return {"status": "PASS_SYNTHETIC_ONLY_NO_MARKET_SCORES", "batch_size_Q_max_gap": batching_gap,
            "common_sample_index_sha256": receipt_first["sample_indices_sha256_int64_little_endian"],
            "direct_gather_max_gap": max(receipt_first["direct_gather_first_four_draw_Q_max_gap"], receipt_second["direct_gather_first_four_draw_Q_max_gap"]),
            "all_zero_denominator_rejected": rejected,
            "partial_zero_denominator_invalid_draws": partial_receipt["invalid_zero_rw_draws"],
            "market_data_reads": 0, "model_fits": 0, "market_scores": 0}


def write_csv(path, rows, fieldnames):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def same_primary_online_strategy_diagnostic(metric_rows):
    """Post-result point attribution, outside the frozen inferential families.

    The primary's policy was already selected from validation. Use that same
    named policy for each control to distinguish independently tuned pipeline
    comparisons from the gate mechanism's incremental contribution. The policy
    is never selected by a test score, and no new fit or interval is generated.
    """
    by_id = {(row["market"], row["id"]): row for row in metric_rows}
    result = []
    for market in MARKETS:
        main = by_id[market, "validation_selected_matured_readout/"+MAIN]
        strategy = main["strategy"]
        check(strategy in STRATEGIES and main["selected_by_validation"] is True,
              "Supplemental diagnostic policy must be the primary validation choice")
        controls = []
        for family in ("static_response", "amplitude_context", "signed_shock_only"):
            reference = by_id[market, "matured_readout_"+strategy+"/"+family]
            own_selected = by_id[market, "validation_selected_matured_readout/"+family]
            controls.append({"comparator_family": family,
                             "same_policy_reference_id": reference["id"], "same_policy_reference_Q": reference["Q"],
                             "main_Q": main["Q"], "effect_rw_percentage_points": 100*(reference["Q"]-main["Q"]),
                             "relative_Q_improvement_percent": 100*(1-main["Q"]/reference["Q"]) if reference["Q"] > 0 else None,
                             "comparator_independently_validation_selected_strategy": own_selected["strategy"],
                             "comparator_independently_validation_selected_Q": own_selected["Q"],
                             "independently_selected_pipeline_effect_rw_percentage_points": 100*(own_selected["Q"]-main["Q"]),
                             "confidence_interval": None})
        result.append({"market": market, "role": "development", "primary_family": MAIN,
                       "primary_validation_selected_strategy": strategy, "primary_Q": main["Q"],
                       "strategy_chosen_by_test_loss": False, "comparators": controls})
    return {"diagnostic_role": "post_result_descriptive_attribution_requested_after_matured_readout_scores_were_seen",
            "not_a_new_preregistered_test": True, "outside_frozen_enhancement_5_and_mechanism_15_families": True,
            "new_fits": 0, "new_prediction_paths": 0, "new_confidence_intervals": 0,
            "data_source": "All five already executed online policies retained in the frozen experiment",
            "policy_rule": "Use the main model's saved validation-selected policy name for all three comparator families",
            "interpretation": "Different families' independently selected online policies compare equally offered tuning pipelines. Their score difference alone cannot isolate the conditional-duration gate. This point-only same-policy comparison diagnoses that attribution issue without changing the frozen primary comparison.",
            "markets": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/tmp/carbon-primary-model-tricks-runs-20261019f"))
    parser.add_argument("--raw-response-root", type=Path, default=Path("/tmp/carbon-signed-response-runs-20261019e/budget320"))
    parser.add_argument("--raw-classical-root", type=Path, default=Path("/tmp/carbon-signed-response-runs-20261019e/core"))
    parser.add_argument("--datasets", type=Path, default=STUDY/"provenance/RUN_DATASETS.json")
    parser.add_argument("--data-root", type=Path, help="Optional location of the same hash-bound CSV basenames")
    parser.add_argument("--plan", type=Path, default=STUDY/"audits/STATISTICAL_PLAN_BEFORE_RUN.json")
    parser.add_argument("--run-plan", type=Path, help="Optional relocated execution plan; exact original hash required")
    parser.add_argument("--out", type=Path, default=STUDY/"audits")
    parser.add_argument("--batch-size", type=int, default=256, help="Memory batching only; does not change draws or their random stream")
    parser.add_argument("--self-check", action="store_true", help="Synthetic-only numerical check; no files written")
    args = parser.parse_args()
    check(args.batch_size > 0, "Positive batching size")
    if args.self_check:
        print(json.dumps(self_check(), allow_nan=False), flush=True)
        return
    plan = assert_plan(args.plan)
    check(not any((args.out/name).exists() for name in OUTPUT_NAMES), "Use a fresh destination; prior analysis outputs are never overwritten")
    ledger = {}
    tracked(__file__, ledger, "independent_analysis_source")
    tracked(args.plan, ledger, "frozen_statistical_plan")
    synthetic = self_check()
    completions, execution_plan_sha = completion_receipts(args, ledger)
    datasets = read_tracked(args.datasets, ledger, "development_dataset_manifest")["datasets"]
    check(tuple(spec["market"] for spec in datasets) == MARKETS, "Frozen market order")
    all_metrics, all_diagnostics, all_intervals, market_receipts = [], [], [], []
    for spec in datasets:
        data = canonical_data(spec, args, ledger)
        rows, losses, diagnostics, replay = collect_market(data, args, ledger)
        intervals, bootstrap_receipts = paired_statistics(rows, losses, spec["market"], plan, args.batch_size)
        all_metrics.extend(rows)
        all_diagnostics.extend(diagnostics)
        all_intervals.extend(intervals)
        market_receipts.append({**data["metadata"], **replay, "model_or_policy_aggregate_rows": len(rows),
                                "forecast_path_roles_before_selected_policy_aliases": 45,
                                "selected_policy_alias_rows": 4, "frozen_online_paths_replay_raw_baselines": 4,
                                "paired_comparisons_per_block": 28, "bootstrap": bootstrap_receipts})
        primary_points = {row["setting"]: row["Q"] for row in rows if row["family"] == MAIN
                          and row["setting"] in ("raw320",)+ENHANCEMENTS}
        print(json.dumps({"completed_statistics_market": spec["market"], "role": "development",
                          "primary_Q": primary_points, "invalid_rw_draws": sum(row["invalid_zero_rw_draws"] for row in bootstrap_receipts)}), flush=True)
    # Inputs are immutable, and no artifact is written until every market and
    # comparison completes. The ledger contains identities, never private rows.
    for item in ledger.values():
        check(digest(item["path"]) == item["sha256"], "An analysis input changed: " + item["path"])
    check(len(all_metrics) == 98 and len(all_intervals) == 112, "Complete two-market aggregate output")
    args.out.mkdir(parents=True, exist_ok=True)
    common = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "COMPLETE_EXPOSED_DEVELOPMENT_STATISTICS",
              "analysis_role": "Both evaluation tails were scored before phase f; conditional intervals do not undo prior research exposure",
              "statistical_plan_sha256": PLAN_SHA256, "execution_plan_sha256": execution_plan_sha,
              "analysis_source_sha256": digest(__file__), "numpy_version": np.__version__,
              "effect_units": "100*(Q_reference-Q_main), normalized RW percentage points; positive favors the main model",
              "relative_improvement_units": "100*(1-Q_main/Q_reference); improvement in normalized MSE Q, not native RMSE",
              "coverage_scope": "Separate per-market enhancement_5 and mechanism_15 families; no joint market or broader research-search coverage",
              "online_policy_selection_fixed_within_resampling": True,
              "resampling_does_not_retrain_models_or_rerun_online_adaptation": True,
              "aggregate_rows_are_not_independent_experiments": True,
              "forecast_path_accounting": {"metric_rows": 98, "validation_selected_policy_aliases": 8,
                                           "path_roles_before_selected_aliases": 90,
                                           "path_role_breakdown": {"new_ewt": 30, "new_dense_path": 8,
                                                                   "all_online_policies": 40, "reused_raw_references": 12},
                                           "online_frozen_paths_already_replay_reused_raw": 8,
                                           "all_paths_share_the_same_two_market_origin_panels": True},
              "no_winner_is_selected_across_the_five_enhancement_protocols": True,
              "full_series_joint_is_retrospective_and_uses_future_test_prices": True,
              "no_new_p_values_or_universal_publication_threshold": True,
              "no_per_origin_arrays_exported": True}
    dump_new(args.out/OUTPUT_NAMES[0], {**common, "market_receipts": market_receipts,
                                      "metrics": all_metrics, "paired_intervals": all_intervals})
    native_rows = []
    for row in all_metrics:
        for index, horizon in enumerate(H.tolist()):
            native_rows.append({"market": row["market"], "role": row["role"], "model_id": row["id"],
                               "setting": row["setting"], "family": row["family"], "strategy": row["strategy"],
                               "selected_by_validation": row["selected_by_validation"], "alias_of": row["alias_of"],
                               "information_scope": row["information_scope"],
                               "n_origins": row["n_origins"], "horizon_observed_records": horizon, "Q": row["Q"],
                               "MSE": row["mse_by_horizon"][index], "RMSE": row["rmse_by_horizon"][index],
                               "MAE": row["mae_by_horizon"][index], "bias": row["bias_by_horizon"][index],
                               "RW_MSE": row["rw_mse_by_horizon"][index], "MSE_over_RW": row["mse_over_rw_by_horizon"][index]})
    write_csv(args.out/OUTPUT_NAMES[1], native_rows, list(native_rows[0]))
    interval_rows = []
    for row in all_intervals:
        primary = row["pointwise_95_interval_rw_percentage_points"]
        adjusted = row["familywise_95_interval_rw_percentage_points"] or [None, None]
        interval_rows.append({key: row[key] for key in ("market", "role", "block_length", "comparison_id", "comparison_family",
                             "main_id", "reference_id", "information_scope", "main_Q", "reference_Q", "effect_rw_percentage_points",
                             "relative_Q_improvement_percent", "bonferroni_family_size", "valid_draws", "invalid_zero_rw_draws")} |
                             {"pointwise_95_low": primary[0], "pointwise_95_high": primary[1],
                              "familywise_95_low": adjusted[0], "familywise_95_high": adjusted[1]})
    write_csv(args.out/OUTPUT_NAMES[2], interval_rows, list(interval_rows[0]))
    dump_new(args.out/OUTPUT_NAMES[3], {**common, "diagnostics": all_diagnostics})
    dump_new(args.out/OUTPUT_NAMES[4], {**common, **same_primary_online_strategy_diagnostic(all_metrics)})
    artifact_ids = [{"filename": name, "sha256": digest(args.out/name), "bytes": (args.out/name).stat().st_size}
                   for name in OUTPUT_NAMES[:-1]]
    dump_new(args.out/OUTPUT_NAMES[-1], {**common, "synthetic_numerical_check": synthetic,
                                      "completed_runs": completions, "market_receipts": market_receipts,
                                      "input_ledger": list(ledger.values()), "input_hashes_unchanged": True,
                                      "model_or_policy_metric_rows": len(all_metrics), "native_horizon_rows": len(native_rows),
                                      "paired_interval_rows": len(all_intervals), "output_artifacts": artifact_ids,
                                      "limitations": ["Analysis independently reconstructs targets and statistics, but does not independently reimplement model fitting.",
                                                      "Bootstrap resamples realized error paths; it conditions on fitted states and saved validation choices.",
                                                      "All five online policies are descriptive retained candidates; only the validation choice enters declared comparisons.",
                                                      "Full-series EWT information-setting gains are not evidence of deployment-time availability."]})
    print(json.dumps({"status": common["status"], "output_directory": str(args.out.resolve()),
                      "artifacts": list(OUTPUT_NAMES)}), flush=True)


if __name__ == "__main__":
    main()
