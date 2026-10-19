"""Replay all stage-f forecasts from public numeric states and identical data.

This is numerical prediction replay, not a repeat of the 598 offline fits.
It never requires a private stage-e/f PT, NPZ or the original runtime paths.
For online adaptation, the prescribed 13,632 mature-label head solves must be
repeated: nonlinear states remain frozen and strategies are selected on the
replayed validation paths before any replayed test loss is computed.

The public source, plan, exported JSON and dataset identities are checked.
Private arrays go to a new directory outside the repository. No old receipt,
hash, timestamp, trained state or published score is modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
STUDY = SOURCE.parent
REPOSITORY = STUDY.parents[1]
E_STUDY = STUDY.parent / "signed_response_search_20261019e"
sys.path.insert(0, str(SOURCE))

import run_primary_model_tricks as ewt
import signed_response_enhancements as frontend
import run_dense_path_supervision as dense
import run_matured_readout as online

core = ewt.original
MARKETS = online.MARKETS
FAMILIES = online.FAMILIES
HORIZONS = online.HORIZONS
FROZEN_PLAN_SHA = "78893370f6862b1295ba9cfcc5e71fc89776944aeae39346de788a4e09cc8880"
ATOL, RTOL = 1e-10, 1e-9
READ_JSON_FILES = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_public(path):
    path = Path(path).resolve()
    if REPOSITORY not in path.parents or path.suffix != ".json":
        raise ValueError("Only public repository JSON evidence may be loaded")
    content = path.read_bytes()
    READ_JSON_FILES[str(path.relative_to(REPOSITORY))] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def numeric_gap(actual, expected, label):
    left, right = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError(f"Nonfinite value or shape mismatch: {label}")
    gap = float(np.max(np.abs(left - right))) if left.size else 0.
    if not np.allclose(left, right, rtol=RTOL, atol=ATOL):
        raise AssertionError(f"Numeric replay mismatch: {label}; absolute gap={gap}")
    return gap


def load_state(model, receipt):
    state = {key: torch.as_tensor(value, dtype=core.DTYPE) for key, value in receipt["state_dict"].items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise AssertionError("Public numeric state is nonfinite")
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.parameter_counts() != receipt["parameter_counts"]:
        raise AssertionError("Replayed model parameter count differs from its receipt")
    return model


def forecast_model(model, inputs, scale):
    before = online.state_digest(model.state_dict())
    with torch.no_grad():
        prediction = model(torch.as_tensor(inputs, dtype=core.DTYPE)).numpy() * scale
    if online.state_digest(model.state_dict()) != before or not np.isfinite(prediction).all():
        raise AssertionError("Prediction changed a frozen state or is nonfinite")
    return prediction


def ridge_prediction(receipt, features, *, original=False):
    mean_key = "final_column_mean" if original else "column_mean"
    scale_key = "final_column_scale" if original else "column_scale"
    mean, scale = np.asarray(receipt[mean_key]), np.asarray(receipt[scale_key])
    coefficients, intercept = np.asarray(receipt["coefficients"]), np.asarray(receipt["intercept"])
    if mean.shape != (features.shape[1],) or scale.shape != mean.shape or not (scale > 0).all():
        raise AssertionError("Public Ridge scaler shape or positivity changed")
    return ((features - mean) / scale) @ coefficients.T + intercept


def metrics(prediction, observed, variance):
    error = np.asarray(prediction) - observed
    mse, rw = np.mean(error ** 2, axis=0), np.mean(observed ** 2, axis=0)
    if not (rw > 0).all():
        raise AssertionError("RW denominators must be positive")
    return {"mse_by_horizon": mse.tolist(), "rmse_by_horizon": np.sqrt(mse).tolist(),
            "mae_by_horizon": np.mean(np.abs(error), axis=0).tolist(),
            "bias_by_horizon": np.mean(error, axis=0).tolist(),
            "rw_mse_by_horizon": rw.tolist(), "mse_over_rw_by_horizon": (mse / rw).tolist(),
            "mean_mse_over_rw": float(np.mean(mse / rw)),
            "fixed_initial_variance_weighted_loss": float(np.mean(error ** 2 / variance))}


def compare_summary(prediction, d, expected, label):
    observed = d["y"][d["masks"]["test"]]
    if expected["horizons"] != list(HORIZONS) or expected["n_origins"] != len(observed):
        raise AssertionError(f"Public evaluation panel changed: {label}")
    calculated = metrics(prediction, observed, d["variance"])
    gaps = {key: numeric_gap(value, expected[key], label + "/" + key)
            for key, value in calculated.items() if key in expected}
    if not {"mse_by_horizon", "rw_mse_by_horizon", "mse_over_rw_by_horizon", "mean_mse_over_rw"}.issubset(gaps):
        raise AssertionError("Required public metrics are missing")
    return calculated, gaps


def save_prediction(folder, prediction, d, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "predictions.npz"
    if path.exists():
        raise FileExistsError("Do not overwrite a replay prediction panel")
    np.savez_compressed(path, predictions=prediction[None], observed=d["y"][d["masks"]["test"]],
                        origins=d["origins"][d["masks"]["test"]], horizons=np.asarray(HORIZONS),
                        deterministic=np.asarray([True]), **extra)
    return {"path": str(path), "sha256": digest(path)}


def verify_public_export(plan_sha):
    receipt = read_public(STUDY / "provenance/RESULT_EXPORT_RECEIPT.json")
    if receipt["status"] != "EXPORTED_BYTE_IDENTICAL" or receipt["plan_sha256"] != plan_sha:
        raise AssertionError("Public export is not a completed frozen-plan export")
    counts = {}
    for run, expected_count in (("ewt", 557), ("dense_path", 157), ("matured_readout", 38)):
        records = receipt["runs"][run]["files"]
        if len(records) != expected_count or len({row["path"] for row in records}) != expected_count:
            raise AssertionError(f"Public export inventory is incomplete: {run}")
        for row in records:
            path = (STUDY / row["path"]).resolve()
            if (STUDY / "results" / run).resolve() not in path.parents or path.suffix != ".json":
                raise AssertionError("Export path escapes its public run")
            if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise AssertionError(f"Exported JSON identity changed: {row['path']}")
        complete = read_public(STUDY / "results" / run / "EXECUTION_COMPLETE.json")
        protocol = read_public(STUDY / "results" / run / "PROTOCOL_BEFORE_RUN.json")
        if complete["status"] != "completed" or protocol["plan_sha256"] != plan_sha:
            raise AssertionError(f"An original run is incomplete: {run}")
        counts[run] = len(records)
    return counts


def verify_setup(args):
    if digest(args.plan) != args.expected_plan_sha or args.expected_plan_sha != FROZEN_PLAN_SHA:
        raise AssertionError("This adapter replays the fixed stage-f experiment only")
    plan = read_public(args.plan)
    if ewt.source_hashes() != plan["source_sha256"] or ewt.scientific_configuration() != plan["scientific_configuration"]:
        raise AssertionError("Frozen EWT sources or scientific settings changed")
    dependencies = ewt.verify_dependencies()
    if dense.contract() != plan["dense_path"] or online.source_hashes() != plan["matured_readout"]["source_sha256"]:
        raise AssertionError("Frozen dense-path or online source contract changed")
    if tuple(plan["matured_readout"]["strategies_in_tie_order"]) != online.STRATEGIES:
        raise AssertionError("Online strategy order changed")
    exported_counts = verify_public_export(args.expected_plan_sha)
    manifest_path = STUDY / "provenance/RUN_DATASETS.json"
    if digest(manifest_path) != plan["datasets_sha256"]:
        raise AssertionError("Frozen dataset manifest changed")
    manifest = read_public(manifest_path)
    specs = copy.deepcopy(manifest["datasets"])
    if tuple(spec["market"] for spec in specs) != MARKETS:
        raise AssertionError("Both original markets in original order are required")
    for spec in specs:
        spec["file"] = str((args.data_root / (spec["market"] + ".csv")).resolve())
        if spec["role"] != "development" or digest(spec["file"]) != spec["sha256"]:
            raise AssertionError("Canonical bytes or exposed-data role changed")
    return plan, specs, dependencies, exported_counts


def verify_manifest(d, published):
    for key in ("row_boundaries", "counts", "origin_ranges", "training_rms", "train_target_second_moment"):
        if d["manifest"][key] != published[key]:
            raise AssertionError(f"Dataset preparation differs: {d['manifest']['market']}/{key}")


def upstream_response_states(plan, market, family):
    binding = next(row for row in plan["matured_readout"]["checkpoints"]
                   if row["market"] == market and row["family"] == family)
    relative = Path(f"{market}_{core.FOLD}") / family
    folder = E_STUDY / "results/budget320" / relative
    selection = read_public(folder / "SELECTION.json")
    choice = selection["selected"]
    selection_receipt = read_public(folder / choice["receipt"])
    refit_receipt = read_public(folder / "REFIT.json")
    for name in ("SELECTION.json", choice["receipt"], "REFIT.json"):
        if digest(folder / name) != binding["files"][str(relative / name)]:
            raise AssertionError("Public upstream numeric receipt does not match the frozen binding")
    if (online.state_digest(selection_receipt["state_dict"]) != binding["selection_numeric_state_sha256"]
            or online.state_digest(refit_receipt["state_dict"]) != binding["final_numeric_state_sha256"]):
        raise AssertionError("Frozen numeric state digest differs")
    if choice["alpha"] != binding["alpha"] or choice["selected_epoch"] != binding["original_selected_epoch"]:
        raise AssertionError("Old alpha or selected epoch changed")
    return {"selection": online.load_frozen_model(family, selection_receipt),
            "final": online.load_frozen_model(family, refit_receipt), "selected": choice,
            "selection_numeric_state_sha256": binding["selection_numeric_state_sha256"],
            "final_numeric_state_sha256": binding["final_numeric_state_sha256"]}


def record_prediction(rows, block, market, family, setting, prediction, d, expected, out, **extra):
    label = "/".join((block, market, setting, family))
    calculated, gaps = compare_summary(prediction, d, expected, label)
    target = out / block / f"{market}_{core.FOLD}" / setting / family
    artifact = save_prediction(target, prediction, d, **extra)
    row = {"block": block, "market": market, "family": family, "setting": setting,
           "role": "replay_of_exposed_development", "information_scope": expected.get("information_scope", "as_original_frozen_setting"),
           "n_origins": len(prediction), "horizons": list(HORIZONS), **calculated,
           "public_metric_absolute_gaps": gaps, "prediction_artifact": artifact}
    write_new(target / "REPLAY_SUMMARY.json", row)
    rows.append(row)


def verify_selected_metadata(selection):
    trials, selected = selection["trials"], selection["selected"]
    key = ((lambda row: (row["validation_loss"], row["alpha"], row["learning_rate"], row["selected_epoch"]))
           if "learning_rate" in selected else (lambda row: (row["validation_loss"], row["alpha"])))
    if min(trials, key=key) != selected:
        raise AssertionError("Public selection does not obey its frozen tie order")
    return selected


def validation_gap(prediction, d, expected, label):
    calculated = float(np.mean((prediction - d["y"][d["masks"]["val"]]) ** 2 / d["variance"]))
    return numeric_gap(calculated, expected, label)


def replay_raw(plan, data, states, rows, out):
    predictions = {}
    for market in MARKETS:
        d = data[market]
        for family in (*FAMILIES, "ridge_price", "rw"):
            folder = STUDY / "results/ewt" / f"{market}_{core.FOLD}" / "raw" / family
            expected = read_public(folder / "SUMMARY.json")
            if family in FAMILIES:
                source = upstream_response_states(plan, market, family)
                states[market, family] = source
                prediction = forecast_model(source["final"], d["x"][d["masks"]["test"]], d["scale"])
            elif family == "ridge_price":
                receipt = read_public(E_STUDY / "results/core" / f"{market}_{core.FOLD}" / family / "SELECTION.json")
                verify_selected_metadata(receipt)
                prediction = ridge_prediction(receipt, d["x"][d["masks"]["test"], -20:, 0], original=True)
            else:
                prediction = np.zeros_like(d["y"][d["masks"]["test"]])
            predictions[market, family] = prediction
            record_prediction(rows, "ewt", market, family, "raw", prediction, d, expected, out)
    return predictions


def replay_ewt(specs, data, rows, checks, out):
    for spec in specs:
        market, d = spec["market"], data[spec["market"]]
        windows = np.stack([frontend.window_changes(d["price"], int(t)) for t in d["origins"]])
        for setting in frontend.NEW_SETTINGS:
            arrays, manifest = frontend.transformed_inputs(d, setting, window_cache=windows)
            root = STUDY / "results/ewt" / f"{market}_{core.FOLD}" / setting
            published = read_public(root / "PREPROCESS_MANIFEST.json")
            for subset in ("train", "val", "final", "test"):
                if manifest["subsets"][subset] != published["subsets"][subset]:
                    raise AssertionError(f"EWT input/target digest or information endpoint differs: {market}/{setting}/{subset}")
            checks.append({"block": "ewt", "market": market, "setting": setting,
                           "all_four_input_and_target_digest_pairs_equal": True})
            for family in (*FAMILIES, "ridge_price"):
                folder = root / family
                selection = read_public(folder / "SELECTION.json")
                selected = verify_selected_metadata(selection)
                val_receipt = read_public(folder / selected["receipt"])
                receipt = read_public(folder / "REFIT.json")
                if family == "ridge_price":
                    features = {subset: arrays[subset][:, -20:, [0, 3, 4, 5]].transpose(0, 2, 1).reshape(len(arrays[subset]), 80)
                                for subset in ("val", "test")}
                    val_prediction = ridge_prediction(val_receipt, features["val"])
                    prediction = ridge_prediction(receipt, features["test"])
                else:
                    val_model = load_state(frontend.StateInputSignedResponse(family), val_receipt)
                    model = load_state(frontend.StateInputSignedResponse(family), receipt)
                    val_prediction = forecast_model(val_model, arrays["val"], d["scale"])
                    prediction = forecast_model(model, arrays["test"], d["scale"])
                gap = validation_gap(val_prediction, d, selected["validation_loss"], f"ewt/{market}/{setting}/{family}/validation")
                checks.append({"block": "ewt", "market": market, "setting": setting, "family": family,
                               "selected_validation_loss_absolute_gap": gap,
                               "selection_metadata_order_verified": True})
                record_prediction(rows, "ewt", market, family, setting, prediction, d,
                                  read_public(folder / "SUMMARY.json"), out)


def replay_dense(specs, data, rows, checks, out):
    for spec in specs:
        market = spec["market"]
        d = dense.dense_data(spec)
        folder = STUDY / "results/dense_path" / f"{market}_{core.FOLD}"
        manifest = read_public(folder / "INPUT_MANIFEST.json")
        verify_manifest(d, manifest)
        if d["dense_variance"].tolist() != manifest["dense_train_target_second_moment"]:
            raise AssertionError("Dense-path training target second moments changed")
        for family in FAMILIES:
            root = folder / family
            selected = verify_selected_metadata(read_public(root / "SELECTION.json"))
            val_receipt = read_public(root / selected["receipt"])
            receipt = read_public(root / "REFIT.json")
            val_model = load_state(dense.SignedResponse(family, horizons=dense.TRAIN_HORIZONS), val_receipt)
            model = load_state(dense.SignedResponse(family, horizons=dense.TRAIN_HORIZONS), receipt)
            val_path = forecast_model(val_model, d["x"][d["masks"]["val"]], d["scale"])
            test_path = forecast_model(model, d["x"][d["masks"]["test"]], d["scale"])
            gap = validation_gap(val_path[:, dense.EVAL_INDICES], d, selected["validation_loss"],
                                 f"dense_path/{market}/{family}/validation")
            checks.append({"block": "dense_path", "market": market, "family": family,
                           "selected_validation_loss_absolute_gap": gap,
                           "selection_metadata_order_verified": True, "training_horizons": list(dense.TRAIN_HORIZONS)})
            record_prediction(rows, "dense_path", market, family, "raw_dense_path_supervision",
                              test_path[:, dense.EVAL_INDICES], d, read_public(root / "SUMMARY.json"), out,
                              dense_path_predictions=test_path[None])


def replay_online(data, states, raw_predictions, rows, checks, out):
    total_solves = 0
    for market in MARKETS:
        d = data[market]
        for family in FAMILIES:
            source = states[market, family]
            root = STUDY / "results/matured_readout" / f"{market}_{core.FOLD}" / family
            published_selection = read_public(root / "SELECTION.json")
            alpha = source["selected"]["alpha"]
            val_trace = online.forecast_paths(d, source["selection"], d["masks"]["val"], alpha, d["masks"]["train"])
            observed_val = d["y"][d["masks"]["val"]]
            losses = np.mean((val_trace["predictions"] - observed_val[None]) ** 2 / d["variance"][None, None], axis=(1, 2))
            if tuple(row["strategy"] for row in published_selection["trials"]) != online.STRATEGIES:
                raise AssertionError("Published online validation strategy order changed")
            loss_gap = numeric_gap(losses, [row["validation_loss"] for row in published_selection["trials"]],
                                   f"online/{market}/{family}/five_validation_losses")
            selected_index = min(range(len(online.STRATEGIES)), key=lambda index: (float(losses[index]), index))
            selected = online.STRATEGIES[selected_index]
            if selected != published_selection["selected_strategy"] or published_selection["test_loss_used_for_selection"]:
                raise AssertionError("Replayed validation choice differs from the pre-test public choice")
            dest = out / "matured_readout" / f"{market}_{core.FOLD}" / family
            # This record is created before this replay's test forecasting and scoring.
            write_new(dest / "VALIDATION_SELECTION_BEFORE_TEST_REPLAY.json", {
                "market": market, "family": family, "selected_strategy": selected,
                "all_strategy_validation_losses": losses.tolist(), "validation_loss_absolute_gap": loss_gap,
                "public_selection_agrees": True, "test_loss_used_for_replay_selection": False})
            test_trace = online.forecast_paths(d, source["final"], d["masks"]["test"], alpha, d["masks"]["final"])
            frozen_gap = numeric_gap(test_trace["predictions"][0], raw_predictions[market, family],
                                     f"online/{market}/{family}/frozen_vs_raw_prediction")
            published_rows = read_public(root / "ALL_STRATEGY_SUMMARIES.json")
            if tuple(row["strategy"] for row in published_rows) != online.STRATEGIES:
                raise AssertionError("Public online results omit or reorder a declared strategy")
            for index, strategy in enumerate(online.STRATEGIES):
                expected = published_rows[index]
                if expected["selected_by_validation"] != (index == selected_index):
                    raise AssertionError("Public online selected-row flag differs")
                record_prediction(rows, "matured_readout", market, family, strategy,
                                  test_trace["predictions"][index], d, expected, out)
            payload = {"strategies": np.asarray(online.STRATEGIES), "horizons": np.asarray(HORIZONS),
                       "selected_strategy_index": np.asarray(selected_index),
                       "validation_observed": observed_val, "test_observed": d["y"][d["masks"]["test"]]}
            for prefix, trace in (("validation", val_trace), ("test", test_trace)):
                payload.update({prefix + "_" + key: value for key, value in trace.items() if key != "ridge_solves"})
            np.savez_compressed(dest / "paths.npz", **payload)
            solves = val_trace["ridge_solves"] + test_trace["ridge_solves"]
            total_solves += solves
            checks.append({"block": "matured_readout", "market": market, "family": family,
                           "all_five_validation_loss_absolute_gap": loss_gap,
                           "selected_strategy": selected, "public_selection_agrees": True,
                           "frozen_prediction_absolute_gap": frozen_gap, "matured_readout_solves": solves,
                           "maximum_normal_equation_residual": float(max(val_trace["normal_equation_residual"].max(),
                                                                          test_trace["normal_equation_residual"].max()))})
    if total_solves != 13632:
        raise AssertionError("Unexpected online readout solve count")
    return total_solves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing the two identical canonical MARKET.csv files")
    parser.add_argument("--out", type=Path, required=True, help="New private replay output outside the repository")
    parser.add_argument("--plan", type=Path, default=STUDY / "audits/EXPERIMENT_PLAN_BEFORE_RUN.json")
    parser.add_argument("--expected-plan-sha", default=FROZEN_PLAN_SHA)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check public/data/source identities only; no model forward, solve or score")
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = args.out.resolve()
    if out == REPOSITORY or REPOSITORY in out.parents:
        raise ValueError("All replay arrays must stay outside the public repository")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Use a new empty replay directory")
    plan, specs, dependencies, export_counts = verify_setup(args)
    identity = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "adapter_sha256": digest(__file__), "original_plan_sha256": args.expected_plan_sha,
                "dependencies": dependencies, "datasets": specs, "runtime": core.runtime(),
                "public_json_inventory_verified": export_counts,
                "scope": "Numerical prediction replay from public self-trained JSON states. No old PT/NPZ bytes or original private paths are required. Original receipts remain unchanged; this run has a new identity.",
                "offline_selection_or_refit_runs": 0, "nonlinear_optimizer_steps": 0,
                "expected_deterministic_prediction_panels": 90, "expected_online_readout_solves": 13632,
                "comparison_tolerance": {"absolute": ATOL, "relative": RTOL},
                "information_settings_and_development_roles_retained": True}
    if args.preflight_only:
        write_new(out / "PREFLIGHT.json", {**identity, "status": "PASS_PUBLIC_IDENTITY_NO_FORECAST",
                                          "model_forward_calls": 0, "new_head_solves": 0, "forecast_scores": 0})
        print("PASS_PUBLIC_IDENTITY_NO_FORECAST", flush=True)
        return
    write_new(out / "REPLAY_PROTOCOL_BEFORE_PREDICTION.json", identity)
    data = {spec["market"]: core.prepare(spec) for spec in specs}
    for market in MARKETS:
        for run in ("ewt", "dense_path", "matured_readout"):
            verify_manifest(data[market], read_public(STUDY / "results" / run / f"{market}_{core.FOLD}" / "INPUT_MANIFEST.json"))
    rows, checks, states = [], [], {}
    raw_predictions = replay_raw(plan, data, states, rows, out)
    print("Replayed 12 raw baseline panels from public numeric states", flush=True)
    replay_ewt(specs, data, rows, checks, out)
    print("Replayed all 30 EWT panels and selected validation states", flush=True)
    replay_dense(specs, data, rows, checks, out)
    print("Replayed all 8 dense-path panels and selected validation states", flush=True)
    total_solves = replay_online(data, states, raw_predictions, rows, checks, out)
    print("Replayed all 40 mature-label strategies; validation selections agree", flush=True)
    if len(rows) != 90:
        raise AssertionError("The full declared prediction inventory was not replayed")
    if (digest(__file__) != identity["adapter_sha256"] or digest(args.plan) != args.expected_plan_sha
            or ewt.verify_dependencies() != dependencies or ewt.source_hashes() != plan["source_sha256"]
            or dense.contract() != plan["dense_path"] or online.source_hashes() != plan["matured_readout"]["source_sha256"]):
        raise AssertionError("An adapter, model source, dependency or plan changed during replay")
    if any(digest(spec["file"]) != spec["sha256"] for spec in specs):
        raise AssertionError("Canonical data changed during replay")
    if any(digest(REPOSITORY / relative) != sha for relative, sha in READ_JSON_FILES.items()):
        raise AssertionError("A consumed public receipt changed during replay")
    write_new(out / "ALL_REPLAY_SUMMARIES.json", rows)
    maximum = max(gap for row in rows for gap in row["public_metric_absolute_gaps"].values())
    receipt = {**identity, "status": "PASS_ALL_PUBLIC_NUMERIC_STATE_REPLAYS",
               "prediction_panels": {"ewt_new": 30, "raw_reused": 12, "dense_path": 8, "online_strategies": 40},
               "total_prediction_panels": len(rows), "online_matured_readout_solves": total_solves,
               "maximum_public_metric_absolute_gap": maximum, "checks": checks,
               "public_json_files_consumed": dict(sorted(READ_JSON_FILES.items())),
               "data_source_plan_and_public_receipts_unchanged": True,
               "full_grid_metadata_selection_checked_without_retraining": True,
               "all_online_validation_choices_recomputed_before_own_test_replay": True,
               "independent_algorithm_reimplementation": False,
               "private_arrays_require_data_rights_and_are_not_exported": True}
    write_new(out / "REPLAY_AUDIT.json", receipt)
    print(json.dumps({"status": receipt["status"], "panels": len(rows),
                      "maximum_public_metric_absolute_gap": maximum,
                      "online_head_solves": total_solves, "offline_fit_runs": 0}), flush=True)


if __name__ == "__main__":
    main()
