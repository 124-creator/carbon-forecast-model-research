"""Frozen chronological classical baselines for the auction-clearing study.

Only development examples are accepted. Training starts only when an explicit
--freeze-id is supplied. --check-only validates interfaces/splits without fitting,
predicting, scoring, or writing artifacts. Individual predictions stay local.
"""
from __future__ import annotations

import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "2"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MODELS = ("RW", "LastBidMedian", "Ridge", "HistGB")
SEED = 17
ATOM_COUNT = 101
QUANTILES = (np.arange(ATOM_COUNT, dtype=np.float64) + 0.5) / ATOM_COUNT


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def prepare_inputs(root, config):
    data_path = root / "local_only/examples_development.npz"
    metadata_path = root / "local_only/example_metadata.csv"
    audit_path = root / "EXAMPLE_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if sha(data_path) != audit["examples_sha256"] or sha(metadata_path) != audit["metadata_sha256"]:
        raise ValueError("Examples or metadata do not match the source-builder audit")
    with np.load(data_path, allow_pickle=False) as archive:
        keys = ["x", "context", "y", "anchor", "scale", "completed", "last_bid_median"]
        data = {key: np.asarray(archive[key], dtype=np.float64) for key in keys}
    meta = pd.read_csv(metadata_path, dtype={"issue_date": str, "target_date": str,
                                           "history_last_date": str, "calendar_available_date": str})
    n = len(meta)
    if n != audit["n_examples"] or any(len(v) != n for v in data.values()):
        raise ValueError("Example arrays and metadata are not aligned")
    if meta.example_id.duplicated().any():
        raise ValueError("Duplicate example identity")
    if not ((meta.target_date >= meta.issue_date).all()
            and (meta.history_last_date < meta.issue_date).all()
            and (meta.calendar_available_date <= meta.issue_date).all()):
        raise ValueError("Causal example date assertions failed")
    if meta.target_date.max() >= "2024-01-01" or meta.issue_date.max() >= "2024-01-01":
        raise ValueError("This runner refuses 2024/2025 confirmation examples")
    if list(config["classical_models"]) != list(MODELS):
        raise ValueError("Classical model list changed from the implemented frozen protocol")
    if config["tuning_seed"] != SEED or config["evaluation_issue_years"] != [2021, 2022, 2023]:
        raise ValueError("Seed or fold contract differs from the implemented protocol")
    if config["horizons"] != [1, 5, 20] or data["x"].shape[1] != config["history_length"]:
        raise ValueError("Horizon or history-length contract differs from configuration")
    if data["x"].shape[2] != len(audit["feature_names"]) or data["context"].shape[1] != len(audit["context_names"]):
        raise ValueError("Feature dimensions differ from the audited schema")
    data["completed"] = data["completed"].astype(bool)
    meta_completed = meta.completed.astype(str).str.lower().eq("true").to_numpy()
    if not np.array_equal(data["completed"], meta_completed):
        raise ValueError("Completed masks disagree")
    if not (np.isfinite(data["y"][data["completed"]]).all()
            and np.isnan(data["y"][~data["completed"]]).all()):
        raise ValueError("Missing/cancelled labels must stay NaN; completed labels must be finite")
    for key in ("x", "context", "anchor", "scale", "last_bid_median"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite known input: {key}; no post-freeze imputation is allowed")
    if not (data["anchor"] > 0).all() or not (data["scale"] > 0).all():
        raise ValueError("Anchor and scale must be positive")
    data["features"] = np.concatenate((data["x"].reshape(n, -1), data["context"]), axis=1)
    data["horizon"] = meta.horizon.to_numpy(dtype=np.int64)
    data["issue_year"] = meta.issue_year.to_numpy(dtype=np.int64)
    return data, meta, audit


def split_indices(meta, completed, year):
    initial_cutoff = f"{year - 1}-01-01"
    final_cutoff = f"{year}-01-01"
    initial = np.flatnonzero(completed & (meta.target_date.to_numpy() < initial_cutoff))
    validation = np.flatnonzero(completed & (meta.issue_year.to_numpy() == year - 1)
                                & (meta.target_date.to_numpy() < final_cutoff))
    final = np.flatnonzero(completed & (meta.target_date.to_numpy() < final_cutoff))
    evaluation = np.flatnonzero(meta.issue_year.to_numpy() == year)
    if any(len(rows) == 0 for rows in (initial, validation, final, evaluation)):
        raise ValueError(f"Empty required split for {year}")
    if np.intersect1d(initial, validation).size or np.intersect1d(final, evaluation).size:
        raise ValueError("Training/validation or final-fit/evaluation split overlap")
    if not set(initial).issubset(final) or not set(validation).issubset(final):
        raise ValueError("Final fit omits already matured eligible labels")
    return {"initial_train": initial, "validation": validation,
            "final_train": final, "evaluation": evaluation}


def candidates(model, config):
    if model in ("RW", "LastBidMedian"):
        return [{}]
    if model == "Ridge":
        return [{"alpha": float(alpha)} for alpha in config["ridge_alphas"]]
    return [{"max_leaf_nodes": int(leaves),
             "learning_rate": float(config["histgb_learning_rate"]),
             "max_iter": int(config["histgb_max_iter"]),
             "min_samples_leaf": int(config["histgb_min_samples_leaf"]),
             "l2_regularization": float(config["histgb_l2_regularization"]),
             "early_stopping": False} for leaves in config["histgb_max_leaf_nodes"]]


def fit_model(model, parameters, features, labels):
    if model in ("RW", "LastBidMedian"):
        return None
    if model == "Ridge":
        estimator = Ridge(alpha=parameters["alpha"], fit_intercept=True,
                          solver="auto", random_state=SEED)
    else:
        estimator = HistGradientBoostingRegressor(**parameters, random_state=SEED)
    fitted = Pipeline([("scaler", StandardScaler()), ("regressor", estimator)])
    fitted.fit(features, labels)
    return fitted


def predict_mean(model, fitted, data, indices):
    if model == "RW":
        result = np.zeros(len(indices), dtype=np.float64)
    elif model == "LastBidMedian":
        result = (data["last_bid_median"][indices] - data["anchor"][indices]) / data["scale"][indices]
    else:
        result = np.asarray(fitted.predict(data["features"][indices]), dtype=np.float64)
    if result.shape != (len(indices),) or not np.isfinite(result).all():
        raise ValueError("A baseline emitted nonfinite or incorrectly shaped means")
    return result


def validation_score(mean_z, data, indices, horizons):
    cells = []
    for horizon in horizons:
        use = data["horizon"][indices] == horizon
        if not use.any():
            raise ValueError(f"No matured validation examples for horizon {horizon}")
        target = data["y"][indices][use]
        scale = data["scale"][indices][use]
        error = (mean_z[use] - target) * scale
        rw_error = -target * scale
        mse, rw_mse = float(np.mean(error ** 2)), float(np.mean(rw_error ** 2))
        if rw_mse <= 0:
            raise ValueError("RW validation MSE denominator is nonpositive")
        cells.append({"horizon": int(horizon), "n_completed": int(use.sum()),
                      "price_mse": mse, "rw_price_mse": rw_mse, "mse_ratio_to_rw": mse / rw_mse})
    return float(np.mean([c["mse_ratio_to_rw"] for c in cells])), cells


def residual_atoms(mean_z, data, indices, horizons):
    result = {}
    for horizon in horizons:
        use = data["horizon"][indices] == horizon
        residual = data["y"][indices][use] - mean_z[use]
        quantile_atoms = np.quantile(residual, QUANTILES, method="linear")
        quantile_atoms -= quantile_atoms.mean()
        if not np.isfinite(quantile_atoms).all() or abs(quantile_atoms.mean()) > 1e-10:
            raise ValueError("Invalid centered validation-residual distribution")
        result[horizon] = quantile_atoms
    return result


def score_evaluation(mean_z, atoms_z, data, indices, year, horizons):
    """Exact finite-atom CRPS; every score is in original price units."""
    cells = []
    pair_coefficients = (2 * np.arange(1, ATOM_COUNT + 1) - ATOM_COUNT - 1) / ATOM_COUNT ** 2
    interval_lower_index = int(np.searchsorted(np.arange(1, ATOM_COUNT + 1) / ATOM_COUNT, 0.1))
    interval_upper_index = int(np.searchsorted(np.arange(1, ATOM_COUNT + 1) / ATOM_COUNT, 0.9))
    for horizon in horizons:
        issued = data["horizon"][indices] == horizon
        use = issued & data["completed"][indices]
        if not use.any():
            raise ValueError(f"Empty evaluation cell {year}, horizon {horizon}")
        target, scale = data["y"][indices][use], data["scale"][indices][use]
        error = (mean_z[use] - target) * scale
        rw_error = -target * scale
        ordered_atoms = np.sort(atoms_z[use], axis=1)
        crps_z = np.abs(ordered_atoms - target[:, None]).mean(axis=1) - ordered_atoms @ pair_coefficients
        if (crps_z < -1e-9).any():
            raise ValueError("Negative finite-atom CRPS")
        lower, upper = ordered_atoms[:, interval_lower_index], ordered_atoms[:, interval_upper_index]
        mse, rw_mse = float(np.mean(error ** 2)), float(np.mean(rw_error ** 2))
        if rw_mse <= 0:
            raise ValueError("RW evaluation MSE denominator is nonpositive")
        cells.append({"issue_year": int(year), "horizon": int(horizon),
                      "n_issued": int(issued.sum()), "n_completed": int(use.sum()),
                      "n_unscored": int(issued.sum() - use.sum()),
                      "price_mse": mse, "rw_price_mse": rw_mse, "mse_ratio_to_rw": mse / rw_mse,
                      "price_mae": float(np.mean(np.abs(error))),
                      "price_crps": float(np.mean(crps_z * scale)),
                      "interval80_coverage": float(np.mean((target >= lower) & (target <= upper))),
                      "interval80_mean_width": float(np.mean((upper - lower) * scale)),
                      "nonpositive_mean_forecasts_issued": int(np.sum(
                          data["anchor"][indices][issued] + data["scale"][indices][issued] * mean_z[issued] <= 0))})
    return cells


def inventory(path, study_root):
    return {"path": str(path.relative_to(study_root)), "bytes": path.stat().st_size, "sha256": sha(path)}


def run_fold_model(model, year, config, data, meta, audit, splits, local_dir, public_dir, root, input_hashes):
    started = time.perf_counter()
    validation_indices = splits["validation"]
    parameter_grid = candidates(model, config)
    candidate_means, candidate_traces, fitted_candidates = [], [], []
    for candidate_index, parameters in enumerate(parameter_grid):
        fit_started = time.perf_counter()
        fitted = fit_model(model, parameters, data["features"][splits["initial_train"]],
                           data["y"][splits["initial_train"]])
        fit_seconds = time.perf_counter() - fit_started
        validation_mean = predict_mean(model, fitted, data, validation_indices)
        score, cells = validation_score(validation_mean, data, validation_indices, config["horizons"])
        candidate_means.append(validation_mean)
        fitted_candidates.append(fitted)
        candidate_traces.append({"candidate_index": candidate_index, "parameters": parameters,
                                 "initial_fit_seconds": fit_seconds,
                                 "validation_equal_horizon_mse_ratio": score, "validation_cells": cells})
    selected_index = min(range(len(candidate_traces)),
                         key=lambda i: (candidate_traces[i]["validation_equal_horizon_mse_ratio"], i))
    selected_parameters = parameter_grid[selected_index]
    selected_validation_mean = candidate_means[selected_index]
    calibration = residual_atoms(selected_validation_mean, data, validation_indices, config["horizons"])
    # The selected initial fit supplies validation residuals. It is never refit
    # on validation labels before its residual distribution is constructed.
    final_started = time.perf_counter()
    final_model = fit_model(model, selected_parameters, data["features"][splits["final_train"]],
                            data["y"][splits["final_train"]])
    final_fit_seconds = time.perf_counter() - final_started
    evaluation_indices = splits["evaluation"]
    mean_z = predict_mean(model, final_model, data, evaluation_indices)
    atoms_z = np.stack([mean_z[i] + calibration[int(h)]
                       for i, h in enumerate(data["horizon"][evaluation_indices])])
    weights = np.full(atoms_z.shape, 1.0 / ATOM_COUNT, dtype=np.float64)
    if not np.allclose(np.sum(atoms_z * weights, axis=1), mean_z, rtol=1e-12, atol=1e-9):
        raise ValueError("Forecast distribution mean differs from the point forecast")
    if not np.isfinite(atoms_z).all() or not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError("Invalid forecast atoms or weights")
    cells = score_evaluation(mean_z, atoms_z, data, evaluation_indices, year, config["horizons"])
    stem = f"{model}_{year}"
    prediction_path = local_dir / f"{stem}.npz"
    np.savez_compressed(prediction_path,
                        example_indices=evaluation_indices.astype(np.int64), mean_z=mean_z,
                        atoms_z=atoms_z, weights=weights, y_z=data["y"][evaluation_indices],
                        anchor=data["anchor"][evaluation_indices], scale=data["scale"][evaluation_indices],
                        completed=data["completed"][evaluation_indices], horizon=data["horizon"][evaluation_indices],
                        validation_indices=validation_indices.astype(np.int64),
                        validation_mean_z=selected_validation_mean, validation_y_z=data["y"][validation_indices],
                        validation_scale=data["scale"][validation_indices],
                        validation_horizon=data["horizon"][validation_indices],
                        validation_candidate_mean_z=np.stack(candidate_means),
                        selected_candidate_index=np.asarray(selected_index, dtype=np.int64),
                        calibration_horizons=np.asarray(config["horizons"], dtype=np.int64),
                        calibration_atoms_z=np.stack([calibration[h] for h in config["horizons"]]),
                        calibration_quantile_levels=QUANTILES,
                        initial_train_indices=splits["initial_train"].astype(np.int64),
                        final_train_indices=splits["final_train"].astype(np.int64),
                        seed=np.asarray(SEED, dtype=np.int64))
    fitted_path = local_dir / f"{stem}.joblib"
    joblib.dump({"model": model, "year": year, "seed": SEED, "parameters": selected_parameters,
                 "selected_initial_pipeline": fitted_candidates[selected_index], "final_pipeline": final_model,
                 "analytic_rule": "mean_z=0" if model == "RW" else
                                  "mean_z=(last_bid_median-anchor)/scale" if model == "LastBidMedian" else None,
                 "calibration_atoms_by_horizon": calibration,
                 "feature_layout": "row-major flattened x history followed by context",
                 "history_feature_names": audit["feature_names"], "context_names": audit["context_names"],
                 "input_hashes": input_hashes}, fitted_path, compress=3)
    record_path = local_dir / f"{stem}_fit_record.json"
    write_json(record_path, {"model": model, "year": year, "seed": SEED,
                            "input_hashes": input_hashes, "parameters": selected_parameters,
                            "train_label": "standardized residual y; completed and matured labels only",
                            "fit_uses_price_labels": model not in ("RW", "LastBidMedian"),
                            "splits": {name: {"indices": ids.tolist(),
                                              "example_ids": meta.iloc[ids].example_id.tolist()}
                                       for name, ids in splits.items()},
                            "candidate_parameters": parameter_grid, "selected_candidate_index": selected_index})
    trace = {"model": model, "issue_year": year, "seed": SEED,
             "split_counts": {name: len(ids) for name, ids in splits.items()},
             "n_features": int(data["features"].shape[1]),
             "initial_train_target_date_max": str(meta.iloc[splits["initial_train"]].target_date.max()),
             "final_train_target_date_max": str(meta.iloc[splits["final_train"]].target_date.max()),
             "candidate_traces": candidate_traces, "selected_candidate_index": selected_index,
             "selected_parameters": selected_parameters,
             "selection_reason": "Fixed baseline; no tunable parameter" if len(parameter_grid) == 1 else
                                 "Minimum equal-horizon raw-price MSE/RW-MSE on the preceding matured validation year; ties use configuration order",
             "calibration": {"source": "selected initial-fit validation residuals, separately by horizon",
                             "atom_count": ATOM_COUNT, "quantile_grid": "(i+0.5)/101, i=0,...,100",
                             "quantile_method": "linear", "center": "subtract mean of 101 quantile atoms",
                             "weights": "101 equal weights", "holdout_fitted": False},
             "final_fit_seconds": final_fit_seconds, "total_seconds": time.perf_counter() - started,
             "evaluation_cells": cells,
             "private_artifacts": [inventory(p, root) for p in (prediction_path, fitted_path, record_path)]}
    write_json(public_dir / f"{stem}_trace.json", trace)
    print(json.dumps({"model": model, "issue_year": year, "status": "completed",
                      "selected_parameters": selected_parameters,
                      "validation_ratio": candidate_traces[selected_index]["validation_equal_horizon_mse_ratio"],
                      "evaluation_equal_horizon_ratio": float(np.mean([c["mse_ratio_to_rw"] for c in cells])),
                      "seconds": trace["total_seconds"]}), flush=True)
    return trace


def summarize(traces, config):
    rw_cells = {(c["issue_year"], c["horizon"]): c
                for trace in traces if trace["model"] == "RW" for c in trace["evaluation_cells"]}
    summaries = {}
    for model in MODELS:
        cells = [dict(cell) for trace in traces if trace["model"] == model for cell in trace["evaluation_cells"]]
        if len(cells) != len(config["evaluation_issue_years"]) * len(config["horizons"]):
            raise ValueError("Incomplete year-horizon metric grid")
        for cell in cells:
            rw = rw_cells[(cell["issue_year"], cell["horizon"])]
            if cell["n_completed"] != rw["n_completed"] or rw["price_crps"] <= 0:
                raise ValueError("Models do not share valid RW scoring denominators")
            cell["crps_ratio_to_validation_calibrated_rw"] = cell["price_crps"] / rw["price_crps"]
        summaries[model] = {
            "primary_equal_cell_mse_ratio_to_rw": float(np.mean([c["mse_ratio_to_rw"] for c in cells])),
            "secondary_equal_cell_crps_ratio_to_rw": float(np.mean([c["crps_ratio_to_validation_calibrated_rw"] for c in cells])),
            "equal_cell_price_mae": float(np.mean([c["price_mae"] for c in cells])),
            "equal_cell_interval80_coverage": float(np.mean([c["interval80_coverage"] for c in cells])),
            "equal_cell_interval80_width": float(np.mean([c["interval80_mean_width"] for c in cells])),
            "cells_beating_rw_mse": int(sum(c["mse_ratio_to_rw"] < 1 for c in cells)),
            "by_year_mse_ratio": {str(year): float(np.mean([c["mse_ratio_to_rw"] for c in cells if c["issue_year"] == year]))
                                  for year in config["evaluation_issue_years"]},
            "cells": cells,
        }
    return {"metric_definition": "Arithmetic mean over 3 issue years x 3 horizons; errors in price units are (mean_z-y_z)*scale",
            "completed_only_scoring": True, "all_issued_forecasts_saved": True,
            "refit_seed": SEED, "classical_fits_are_not_seed_ensembles": True,
            "models": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--freeze-id", help="Explicit root-issued FREEZE identifier; recorded without modifying the protocol")
    parser.add_argument("--check-only", action="store_true", help="No fitting, prediction, scoring, or writes")
    args = parser.parse_args()
    if not args.check_only and (not args.freeze_id or "FREEZE" not in args.freeze_id.upper()):
        parser.error("A root-issued --freeze-id containing FREEZE is required for real training")
    root = args.study_dir.resolve()
    config_path = root / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data, meta, audit = prepare_inputs(root, config)
    splits_by_year = {year: split_indices(meta, data["completed"], year)
                      for year in config["evaluation_issue_years"]}
    input_paths = {"config": config_path, "examples": root / "local_only/examples_development.npz",
                   "metadata": root / "local_only/example_metadata.csv", "example_audit": root / "EXAMPLE_AUDIT.json",
                   "runner": Path(__file__).resolve()}
    input_hashes = {name: sha(path) for name, path in input_paths.items()}
    if args.check_only:
        print(json.dumps({"status": "interface_and_splits_checked_without_training_or_scoring",
                          "n_examples": len(meta), "n_features": int(data["features"].shape[1]),
                          "split_counts": {str(year): {name: len(ids) for name, ids in splits.items()}
                                           for year, splits in splits_by_year.items()},
                          "input_hashes": input_hashes}), flush=True)
        return
    public_dir, local_dir = root / "results/classical", root / "local_only/classical"
    for directory in (public_dir, local_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run evidence: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    receipt_path = public_dir / "CLASSICAL_RUN_RECEIPT.json"
    receipt = {"status": "running", "started_utc": utc_now(), "freeze_id": args.freeze_id,
               "input_hashes": input_hashes,
               "runtime": {"python_executable": sys.executable, "python": platform.python_version(),
                           "numpy": np.__version__, "sklearn": sklearn.__version__,
                           "joblib": joblib.__version__, "cpu_threads": 2},
               "models": list(MODELS), "years": config["evaluation_issue_years"],
               "seed": SEED, "confirmation_data_read": False, "completed_fold_models": []}
    write_json(receipt_path, receipt)
    traces, started = [], time.perf_counter()
    try:
        with threadpool_limits(limits=2):
            for year in config["evaluation_issue_years"]:
                for model in MODELS:
                    trace = run_fold_model(model, year, config, data, meta, audit, splits_by_year[year],
                                           local_dir, public_dir, root, input_hashes)
                    traces.append(trace)
                    receipt["completed_fold_models"].append({"model": model, "year": year})
                    write_json(receipt_path, receipt)
        # An input mutation during execution invalidates the run and is never
        # silently accepted as if all models used the frozen files.
        if input_hashes != {name: sha(path) for name, path in input_paths.items()}:
            raise ValueError("Frozen input or runner changed while classical baselines were running")
        summary = summarize(traces, config)
        summary["input_hashes"] = input_hashes
        summary_path = public_dir / "CLASSICAL_METRICS.json"
        write_json(summary_path, summary)
        receipt.update({"status": "completed", "finished_utc": utc_now(),
                        "total_seconds": time.perf_counter() - started,
                        "metrics": inventory(summary_path, root),
                        "private_artifacts": [artifact for trace in traces for artifact in trace["private_artifacts"]],
                        "public_traces": [inventory(public_dir / f"{model}_{year}_trace.json", root)
                                          for year in config["evaluation_issue_years"] for model in MODELS]})
        write_json(receipt_path, receipt)
        print(json.dumps({"status": "completed", "total_seconds": receipt["total_seconds"],
                          "primary_ratios": {model: values["primary_equal_cell_mse_ratio_to_rw"]
                                             for model, values in summary["models"].items()},
                          "receipt": str(receipt_path)}), flush=True)
    except Exception as error:
        receipt.update({"status": "failed", "finished_utc": utc_now(),
                        "total_seconds": time.perf_counter() - started,
                        "error_type": type(error).__name__, "error": str(error),
                        "traceback": traceback.format_exc()})
        write_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    main()
