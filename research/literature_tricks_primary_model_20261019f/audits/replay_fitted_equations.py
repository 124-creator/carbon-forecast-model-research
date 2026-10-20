"""Read-only independent NumPy replay of completed stage-f fitted equations.

No Torch model, training runner, sklearn estimator or EWT transform is imported.
The EWT front end is read from identity-checked private caches; the response
basis, weighted VP head, Ridge equation, losses and saved predictions are
recomputed independently. This verifies recorded checkpoints, not an Adam
trajectory or global optimality. It never selects a new configuration.

Only complete EWT and dense-path runs may be read. Output contains aggregate
errors and one row per model/setting, never per-origin data or predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
FAMILIES = ("signed_context_response", "static_response", "amplitude_context", "signed_shock_only")
SETTINGS = ("window_past", "train_joint_test_window", "full_series_joint")
EVAL_HORIZONS = (1, 5, 20)
DENSE_HORIZONS = tuple(range(1, 21))
LOOKBACK = 64
FOLD = "fixed602020"
ATOL = 1e-9
RTOL = 1e-8
KKT_TOL = 1e-9


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    header = f"{value.dtype.str}:{value.shape}:".encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        if isinstance(value, str):
            handle.write(value)
        else:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")


class Checks:
    def __init__(self):
        self.maxima = {}
        self.counts = Counter()
        self.failures = []

    def close(self, name, actual, expected, context, *, atol=ATOL, rtol=RTOL):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if actual.shape != expected.shape:
            raise AssertionError(f"{context}: {name} shape {actual.shape} != {expected.shape}")
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(f"{context}: {name} is not finite")
        gap = float(np.max(np.abs(actual-expected), initial=0.))
        scaled = float(np.max(np.abs(actual-expected)/(1.+np.abs(expected)), initial=0.))
        record = self.maxima.setdefault(name, {"max_absolute_gap": 0., "max_gap_over_one_plus_reference_abs": 0.,
                                               "comparisons": 0})
        record["max_absolute_gap"] = max(record["max_absolute_gap"], gap)
        record["max_gap_over_one_plus_reference_abs"] = max(record["max_gap_over_one_plus_reference_abs"], scaled)
        record["comparisons"] += 1
        if not np.allclose(actual, expected, atol=atol, rtol=rtol):
            self.failures.append({"context": context, "check": name, "max_absolute_gap": gap,
                                  "max_gap_over_one_plus_reference_abs": scaled})
        return gap

    def kkt(self, name, residual, rhs, context):
        value = float(np.max(np.abs(residual), initial=0.))
        relative = value/(1.+float(np.max(np.abs(rhs), initial=0.)))
        record = self.maxima.setdefault(name, {"max_absolute_residual": 0., "max_residual_over_one_plus_rhs_abs": 0.,
                                               "comparisons": 0})
        record["max_absolute_residual"] = max(record["max_absolute_residual"], value)
        record["max_residual_over_one_plus_rhs_abs"] = max(record["max_residual_over_one_plus_rhs_abs"], relative)
        record["comparisons"] += 1
        if relative > KKT_TOL:
            self.failures.append({"context": context, "check": name, "relative_residual": relative})


def canonical_data(spec):
    """Independent origin/maturity, raw feature and label reconstruction."""
    if digest(spec["file"]) != spec["sha256"]:
        raise AssertionError("Canonical data identity changed")
    frame = pd.read_csv(spec["file"])
    if list(frame) != ["date", "price", "volume_tons", "turnover_yuan"]:
        raise AssertionError("Canonical four-column schema changed")
    if not frame.date.is_unique or not frame.date.is_monotonic_increasing:
        raise AssertionError("Canonical dates changed order or uniqueness")
    price = frame.price.to_numpy(float)
    volume = frame.volume_tons.to_numpy(float)
    n = len(price)
    a, b = n*3//5, n*3//5+n//5
    origins = np.arange(LOOKBACK, n-max(EVAL_HORIZONS))
    masks = {"train": origins+20 < a,
             "val": (origins >= a) & (origins+20 < b),
             "final": origins+20 < b,
             "test": origins >= b}
    returns = np.r_[0., np.diff(price)]
    scale = max(float(np.sqrt(np.mean(returns[1:a]**2))), 1e-7)
    valid = np.isfinite(volume) & (volume >= 0)
    logs = np.zeros(n)
    logs[valid] = np.log1p(volume[valid])
    context = np.zeros((n, 2))
    for t in range(n):
        past_valid = valid[max(0, t-60):t]
        if valid[t] and past_valid.sum() >= 10:
            past = logs[max(0, t-60):t][past_valid]
            context[t] = [np.clip((logs[t]-np.median(past))/max(float(np.std(past)), .25), -5, 5), 1.]
    tokens = np.column_stack((returns/scale, context))
    x = np.stack([tokens[t-LOOKBACK+1:t+1] for t in origins])
    y = np.column_stack([price[origins+h]-price[origins] for h in EVAL_HORIZONS])
    dense_y = np.column_stack([price[origins+h]-price[origins] for h in DENSE_HORIZONS])
    variance = np.maximum(np.mean(y[masks["train"]]**2, axis=0), scale**2*1e-6)
    dense_variance = np.maximum(np.mean(dense_y[masks["train"]]**2, axis=0), scale**2*1e-6)
    dense_variance[np.asarray(EVAL_HORIZONS)-1] = variance
    if not np.array_equal(dense_y[:, np.asarray(EVAL_HORIZONS)-1], y):
        raise AssertionError("Dense targets differ at evaluated horizons")
    return {"price": price, "origins": origins, "x": x, "y": y, "dense_y": dense_y,
            "scale": scale, "variance": variance, "dense_variance": dense_variance,
            "masks": masks, "row_boundaries": [0, a, b, n]}


def rates(raw):
    raw = np.asarray(raw, dtype=np.float64)
    first = .98/(1.+np.exp(-raw[..., 0]))
    second = first+(.98-first)/(1.+np.exp(-raw[..., 1]))
    return np.stack((first, second), axis=-1)


def basis_numpy(x, state, family, horizons):
    """Explicit shared-channel signed response, without any model import."""
    x = np.asarray(x, dtype=np.float64)
    if family not in FAMILIES or x.shape[1] != LOOKBACK or x.shape[2] not in (3, 6):
        raise AssertionError("Unexpected response input or family")
    columns = (0,) if x.shape[2] == 3 else (0, 3, 4, 5)
    raw = np.asarray(state["rate_raw"], dtype=np.float64)
    rho = rates(raw)
    gate = np.asarray(state["gate"], dtype=np.float64)
    z = np.column_stack((np.log1p(np.abs(x[:, -1, 0])), x[:, -1, 1]*x[:, -1, 2]))
    z = z[:, :1] if family == "signed_shock_only" else z
    if family == "static_response" and np.any(gate != 0):
        raise AssertionError("The static gate is not its fixed zero buffer")
    q = 2*np.tanh(z @ gate.T)
    future = (rates(raw[None]+q) if family in ("signed_context_response", "signed_shock_only")
              else np.broadcast_to(rho, (len(x), 2)))
    amplitude = np.exp(q) if family == "amplitude_context" else np.ones_like(q)
    history_weights = (1-rho)[None] * rho[None]**np.arange(LOOKBACK-1, -1, -1)[:, None]
    basis = np.zeros((len(x), len(horizons), 1+2*len(columns)))
    basis[:, :, 0] = np.asarray(horizons)[None]
    for channel_index, column in enumerate(columns):
        states = x[:, :, column] @ history_weights
        for hi, h in enumerate(horizons):
            duration = sum(future**j for j in range(1, h+1))
            basis[:, hi, 1+2*channel_index:3+2*channel_index] = states*amplitude*duration
    return basis


def vp_equation(basis, labels, variance, alpha):
    """Independent profiled training equation including penalized drift."""
    weight = 1./np.sqrt(variance)
    unscaled = (basis*weight[None, :, None]).reshape(-1, basis.shape[-1])
    target = (labels*weight[None]).reshape(-1)
    column_scale = np.maximum(np.sqrt(np.mean(unscaled*unscaled, axis=0)), 1e-8)
    design = unscaled/column_scale
    gram = design.T @ design / len(design)
    rhs = design.T @ target / len(design)
    regularized = gram+alpha*np.eye(gram.shape[0])
    standardized = np.linalg.solve(regularized, rhs)
    beta = standardized/column_scale
    loss = float(np.mean((basis @ beta-labels)**2/variance))
    objective = loss+alpha*float(np.sum(standardized**2))
    return beta, column_scale, regularized, rhs, loss, objective


def verify_history(receipt, is_final, checks, context):
    history = receipt["history"]
    if [r["epoch"] for r in history] != list(range(receipt["epochs_executed"]+1)):
        raise AssertionError(f"{context}: nonconsecutive fit history")
    selected_epoch = receipt["selected_epoch"]
    if not 0 <= selected_epoch <= receipt["epochs_executed"] <= receipt["config"]["maximum_epochs"]:
        raise AssertionError(f"{context}: invalid selected/executed epoch")
    gradient_entries = [r for r in history if "gradient_norm_before_clip" in r]
    if len(gradient_entries) != receipt["epochs_executed"]:
        raise AssertionError(f"{context}: update count and gradient history differ")
    positive = sum(r["gradient_norm_before_clip"] > 0 for r in gradient_entries)
    if positive != receipt["positive_gradient_steps"]:
        raise AssertionError(f"{context}: positive gradient count differs")
    if is_final:
        if selected_epoch != history[-1]["epoch"] or receipt["validation_loss"] is not None:
            raise AssertionError(f"{context}: final refit did not retain last state")
    else:
        best, expected_epoch = float("inf"), None
        tolerance = receipt["config"]["selection_tolerance"]
        for row in history:
            if row["validation_loss"] < best-tolerance:
                best, expected_epoch = row["validation_loss"], row["epoch"]
        if selected_epoch != expected_epoch:
            raise AssertionError(f"{context}: selected epoch disagrees with tolerance rule")
        checks.close("selected_validation_history_loss", receipt["validation_loss"], best, context)
    checks.close("selected_history_rates", rates(receipt["state_dict"]["rate_raw"]),
                 history[selected_epoch]["historical_rates"], context)
    initial_raw = np.asarray([math.log(.5/(.98-.5)), math.log((.9-.5)/(.98-.9))])
    state = receipt["state_dict"]
    changes = {"rate_raw": float(np.max(np.abs(np.asarray(state["rate_raw"])-initial_raw)))}
    if receipt["family"] != "static_response":
        changes["gate"] = float(np.max(np.abs(state["gate"])))
    if set(changes) != set(receipt["selected_nonlinear_parameter_max_change"]):
        raise AssertionError(f"{context}: nonlinear parameter identity differs")
    for name, change in changes.items():
        checks.close("selected_parameter_change", change,
                     receipt["selected_nonlinear_parameter_max_change"][name], context)
    return changes


def replay_response(receipt, x_train, y_train, variance, horizons, checks, context,
                    *, x_val=None, y_val=None, eval_variance=None, is_final=False):
    changes = verify_history(receipt, is_final, checks, context)
    state = receipt["state_dict"]
    basis = basis_numpy(x_train, state, receipt["family"], horizons)
    beta, scale, regularized, rhs, loss, objective = vp_equation(
        basis, y_train, variance, receipt["config"]["alpha"])
    saved_beta = np.asarray(state["coefficients"])
    saved_scale = np.asarray(state["head_column_scale"])
    checks.close("response_head_coefficients", saved_beta, beta, context)
    checks.close("response_head_column_scale", saved_scale, scale, context)
    checks.kkt("response_saved_head_kkt", regularized @ (saved_beta*scale)-rhs, rhs, context)
    checks.kkt("response_numpy_solve_kkt", regularized @ (beta*scale)-rhs, rhs, context)
    row = receipt["history"][receipt["selected_epoch"]]
    checks.close("response_training_data_loss", loss, row["training_data_loss"], context)
    checks.close("response_training_profiled_objective", objective, row["training_profiled_objective"], context)
    if x_val is not None:
        prediction = basis_numpy(x_val, state, receipt["family"], EVAL_HORIZONS) @ saved_beta
        validation = float(np.mean((prediction-y_val)**2/eval_variance))
        checks.close("response_validation_loss", validation, receipt["validation_loss"], context)
    return changes


def record_identity(ledger, path, root):
    ledger.append({"path": str(Path(path).relative_to(root)), "sha256": digest(path)})


def verify_selection(selection, dest, checks, context, *, ridge=False):
    trials = selection["trials"]
    expected_count = 8 if ridge else 16
    if len(trials) != expected_count or len({t["receipt"] for t in trials}) != expected_count:
        raise AssertionError(f"{context}: incomplete selection grid")
    keys = ("validation_loss", "alpha") if ridge else ("validation_loss", "alpha", "learning_rate", "selected_epoch")
    chosen = min(trials, key=lambda t: tuple(t[k] for k in keys))
    if chosen != selection["selected"]:
        raise AssertionError(f"{context}: selected trial differs from fixed tie rule")
    for trial in trials:
        receipt = read(dest/trial["receipt"])
        checks.close("trial_table_validation_loss", receipt["validation_loss"], trial["validation_loss"], context)
        if ridge:
            if receipt["alpha"] != trial["alpha"]:
                raise AssertionError(f"{context}: Ridge trial identity mismatch")
        else:
            if any(receipt[k] != trial[k] for k in ("selected_epoch", "epochs_executed", "positive_gradient_steps")):
                raise AssertionError(f"{context}: trial epoch or update identity mismatch")
            if any(receipt["config"][k] != trial[k] for k in ("alpha", "learning_rate")):
                raise AssertionError(f"{context}: response trial configuration mismatch")
    return chosen


def verify_prediction_file(path, prediction, native_y, origins, summary, checks, context, *, dense_path=None):
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["observed"], native_y) or not np.array_equal(z["origins"], origins):
            raise AssertionError(f"{context}: stored target/origin alignment differs")
        if not np.array_equal(z["horizons"], EVAL_HORIZONS) or not np.array_equal(z["deterministic"], [True]):
            raise AssertionError(f"{context}: stored horizon/deterministic identity differs")
        checks.close("native_test_predictions", z["predictions"], prediction[None], context)
        if dense_path is not None:
            checks.close("native_dense_path_predictions", z["dense_path_predictions"], dense_path[None], context)
    mse = np.mean((prediction-native_y)**2, axis=0)
    rw = np.mean(native_y*native_y, axis=0)
    checks.close("summary_mse", summary["mse_by_horizon"], mse, context)
    checks.close("summary_rw_mse", summary["rw_mse_by_horizon"], rw, context)
    checks.close("summary_mse_over_rw", summary["mse_over_rw_by_horizon"], mse/rw, context)
    checks.close("summary_Q", summary["mean_mse_over_rw"], np.mean(mse/rw), context)
    if "rmse_by_horizon" in summary:
        checks.close("summary_rmse", summary["rmse_by_horizon"], np.sqrt(mse), context)
    if "mae_by_horizon" in summary:
        checks.close("summary_mae", summary["mae_by_horizon"], np.mean(np.abs(prediction-native_y), axis=0), context)


def ewt_cache(run, market, setting, d, checks, ledger, root):
    path = run/f"{market}_{FOLD}"/setting/"PREPROCESS_MANIFEST.json"
    record = read(path)
    record_identity(ledger, path, root)
    if record["market"] != market or record["setting"] != setting:
        raise AssertionError("EWT cache identity differs")
    if digest(record["cache_path"]) != record["cache_sha256"]:
        raise AssertionError("EWT cache bytes changed")
    with np.load(record["cache_path"], allow_pickle=False) as z:
        cache = {key: z[key].copy() for key in z.files}
    checks.close("cache_training_scale", cache["scale"], d["scale"], market)
    checks.close("cache_training_variance", cache["variance"], d["variance"], market)
    for subset in ("train", "val", "final", "test"):
        x, y = cache[f"x_{subset}"], cache[f"y_{subset}"]
        item = record["subsets"][subset]
        if array_digest(x) != item["input_sha256"] or array_digest(y) != item["native_target_sha256"]:
            raise AssertionError("EWT cache subset digest differs")
        if not np.array_equal(cache[f"origins_{subset}"], d["origins"][d["masks"][subset]]):
            raise AssertionError("EWT cache maturity mask differs")
        if not np.array_equal(y, d["y"][d["masks"][subset]]):
            raise AssertionError("EWT cache unfiltered labels differ")
        checks.close("cache_raw_three_columns", x[..., :3], d["x"][d["masks"][subset]], market)
    return cache


def response_job(dest, block, market, setting, family, x, labels, variance, eval_variance,
                 native_test_y, test_origins, scale, horizons, checks, ledger, root):
    context = f"{block}/{market}/{setting}/{family}"
    selection = read(dest/"SELECTION.json")
    choice = verify_selection(selection, dest, checks, context)
    record_identity(ledger, dest/"SELECTION.json", root)
    for trial in selection["trials"]:
        path = dest/trial["receipt"]
        receipt = read(path)
        replay_response(receipt, x["train"], labels["train"], variance, horizons, checks,
                        context+"/"+path.name, x_val=x["val"], y_val=labels["val"], eval_variance=eval_variance)
        record_identity(ledger, path, root)
        checks.counts[f"{block}_response_selection_checkpoints"] += 1
        checks.counts["selection_checkpoints_selected_at_cap"] += receipt["selected_epoch"] == receipt["config"]["maximum_epochs"]
        checks.counts["selection_checkpoints_execution_at_cap"] += receipt["epochs_executed"] == receipt["config"]["maximum_epochs"]
    receipt = read(dest/"REFIT.json")
    if any(receipt["config"][k] != choice[k] for k in ("alpha", "learning_rate")):
        raise AssertionError(f"{context}: final alpha or learning rate differs from selected trial")
    if receipt["selected_epoch"] != choice["selected_epoch"] or receipt["epochs_executed"] != choice["selected_epoch"]:
        raise AssertionError(f"{context}: final update budget differs from chosen epoch")
    changes = replay_response(receipt, x["final"], labels["final"], variance, horizons,
                              checks, context+"/REFIT.json", is_final=True)
    record_identity(ledger, dest/"REFIT.json", root)
    checks.counts[f"{block}_response_final_checkpoints"] += 1
    state = receipt["state_dict"]
    path_prediction = basis_numpy(x["test"], state, family, horizons) @ np.asarray(state["coefficients"]) * scale
    prediction = path_prediction if horizons == EVAL_HORIZONS else path_prediction[:, np.asarray(EVAL_HORIZONS)-1]
    summary = read(dest/"SUMMARY.json")
    verify_prediction_file(dest/"predictions.npz", prediction, native_test_y, test_origins,
                           summary, checks, context, dense_path=path_prediction if horizons == DENSE_HORIZONS else None)
    record_identity(ledger, dest/"SUMMARY.json", root)
    record_identity(ledger, dest/"predictions.npz", root)
    selected_steps = sum(t["positive_gradient_steps"] for t in selection["trials"])
    if summary["selection_positive_gradient_steps"] != selected_steps or summary["refit_positive_gradient_steps"] != receipt["positive_gradient_steps"]:
        raise AssertionError(f"{context}: summary update count differs")
    linear_count = len(state["coefficients"])
    nonlinear_count = 2+(0 if family == "static_response" else 2 if family == "signed_shock_only" else 4)
    expected_parameters = {"nonlinear_optimizer_parameters": nonlinear_count,
                           "exactly_fitted_linear_coefficients": linear_count,
                           "total_fitted_scalars": nonlinear_count+linear_count}
    if receipt["parameter_counts"] != expected_parameters or summary["parameter_counts"] != expected_parameters:
        raise AssertionError(f"{context}: parameter counts differ")
    return {"block": block, "market": market, "setting": setting, "family": family,
            "selected_alpha": choice["alpha"], "selected_learning_rate": choice["learning_rate"],
            "selected_epoch": choice["selected_epoch"], "maximum_epochs": receipt["config"]["maximum_epochs"],
            "selected_at_epoch_cap": choice["selected_epoch"] == receipt["config"]["maximum_epochs"],
            "selection_positive_gradient_steps": selected_steps,
            "refit_positive_gradient_steps": receipt["positive_gradient_steps"],
            "final_nonlinear_parameter_max_change": changes,
            "final_any_nonlinear_parameter_changed": any(v > 0 for v in changes.values()),
            "final_linear_coefficients_nonzero": bool(np.any(np.asarray(state["coefficients"]) != 0)),
            "fitted_parameters": nonlinear_count+linear_count}


def ridge_features(x):
    # Explicit channel concatenation is independent of the runner's transpose/
    # reshape implementation. Lag order remains oldest to newest per channel.
    return np.concatenate([x[:, -20:, col] for col in (0, 3, 4, 5)], axis=1)


def replay_ridge(receipt, train, labels, checks, context):
    mean = np.mean(train, axis=0)
    variance = np.var(train, axis=0)
    # sklearn's constant-feature criterion for StandardScaler is based on the
    # floating point error bound of its variance estimator.
    eps = np.finfo(np.float64).eps
    upper = len(train)*eps*variance+(len(train)*mean*eps)**2
    scale = np.sqrt(variance)
    scale[variance <= upper] = 1.
    checks.close("ridge_scaler_mean", receipt["column_mean"], mean, context)
    checks.close("ridge_scaler_scale", receipt["column_scale"], scale, context)
    x = (train-mean)/scale
    x_mean, y_mean = np.mean(x, axis=0), np.mean(labels, axis=0)
    xc, yc = x-x_mean, labels-y_mean
    gram = xc.T @ xc+receipt["alpha"]*np.eye(x.shape[1])
    rhs = xc.T @ yc
    coefficients = np.linalg.solve(gram, rhs).T
    intercept = y_mean-x_mean @ coefficients.T
    checks.close("ridge_coefficients", receipt["coefficients"], coefficients, context)
    checks.close("ridge_intercept", receipt["intercept"], intercept, context)
    checks.kkt("ridge_saved_coefficients_kkt", gram @ np.asarray(receipt["coefficients"]).T-rhs, rhs, context)
    return mean, scale, coefficients, intercept


def ridge_job(dest, market, setting, cache, checks, ledger, root):
    context = f"ewt/{market}/{setting}/ridge_price"
    features = {s: ridge_features(cache[f"x_{s}"]) for s in ("train", "val", "final", "test")}
    selection = read(dest/"SELECTION.json")
    choice = verify_selection(selection, dest, checks, context, ridge=True)
    record_identity(ledger, dest/"SELECTION.json", root)
    for trial in selection["trials"]:
        path = dest/trial["receipt"]
        receipt = read(path)
        mean, scale, coefficients, intercept = replay_ridge(receipt, features["train"], cache["y_train"], checks,
                                                           context+"/"+path.name)
        prediction = ((features["val"]-mean)/scale) @ coefficients.T+intercept
        validation = float(np.mean((prediction-cache["y_val"])**2/cache["variance"]))
        checks.close("ridge_validation_loss", validation, receipt["validation_loss"], context)
        record_identity(ledger, path, root)
        checks.counts["ewt_ridge_selection_checkpoints"] += 1
    receipt = read(dest/"REFIT.json")
    if receipt["alpha"] != choice["alpha"] or receipt["fitted_parameters"] != 243:
        raise AssertionError(f"{context}: final Ridge configuration differs")
    mean, scale, coefficients, intercept = replay_ridge(receipt, features["final"], cache["y_final"], checks,
                                                       context+"/REFIT.json")
    prediction = ((features["test"]-mean)/scale) @ coefficients.T+intercept
    summary = read(dest/"SUMMARY.json")
    verify_prediction_file(dest/"predictions.npz", prediction, cache["y_test"], cache["origins_test"],
                           summary, checks, context)
    for name in ("REFIT.json", "SUMMARY.json", "predictions.npz"):
        record_identity(ledger, dest/name, root)
    checks.counts["ewt_ridge_final_checkpoints"] += 1
    return {"block": "ewt", "market": market, "setting": setting, "family": "ridge_price",
            "selected_alpha": choice["alpha"], "fitted_parameters": 243,
            "selection_checkpoints_replayed": 8, "final_checkpoints_replayed": 1}


def assert_complete(root, plan, plan_sha, checks):
    ewt = read(root/"ewt/EXECUTION_COMPLETE.json")
    dense = read(root/"dense_path/EXECUTION_COMPLETE.json")
    if ewt["status"] != "completed" or dense["status"] != "completed":
        raise AssertionError("Only completed EWT and dense-path runs can be audited")
    if ewt["new_fit_tasks"] != 462 or ewt["new_prediction_sets"] != 30 or ewt["reused_prediction_sets"] != 12:
        raise AssertionError("EWT completion counts differ from frozen design")
    if dense["selection_fits"] != 128 or dense["final_refits"] != 8 or dense["summary_rows"] != 8:
        raise AssertionError("Dense-path completion counts differ from frozen design")
    for block in ("ewt", "dense_path"):
        protocol = read(root/block/"PROTOCOL_BEFORE_RUN.json")
        if protocol["plan_sha256"] != plan_sha:
            raise AssertionError("A private run used a different plan")
        if block == "ewt":
            if protocol["source_sha256"] != plan["source_sha256"] or protocol["scientific_configuration"] != plan["scientific_configuration"]:
                raise AssertionError("EWT frozen configuration differs")
        elif protocol["contract"] != plan["dense_path"]:
            raise AssertionError("Dense-path frozen contract differs")
    return {"ewt": ewt, "dense_path": dense}


def markdown(result):
    lines = ["# 已拟合方程独立 NumPy 复放", "", f"状态：**{result['status']}**。", "",
             "本程序重新计算已完成 checkpoint 的方程、读出、损失和预测；不增加超参数搜索或优化器训练。EWT 的已固定分析特征读取自带 SHA 的私有缓存；原始特征、成熟掩码与 20 步监督标签独立重建。基底、VP 和 Ridge 均使用 NumPy 显式公式，没有调用 Torch 模型前向或 sklearn 估计器。", "",
             "## 覆盖", "", "| 类型 | 选择 checkpoint | 最终 checkpoint |", "|---|---:|---:|",
             f"| EWT 响应 | {result['counts']['ewt_response_selection_checkpoints']} | {result['counts']['ewt_response_final_checkpoints']} |",
             f"| Dense-path 响应 | {result['counts']['dense_response_selection_checkpoints']} | {result['counts']['dense_response_final_checkpoints']} |",
             f"| EWT Ridge | {result['counts']['ewt_ridge_selection_checkpoints']} | {result['counts']['ewt_ridge_final_checkpoints']} |", "",
             "另核验全部选择表的固定 tie 顺序，逐 checkpoint 的 epoch 容差选择、梯度更新计数、最终拟合步数及实际数值参数变化。Dense-path 的训练方差为原始训练期 H1–20 二阶矩；选择与主评价使用原始 H1/H5/H20 方差，未以 20 步平均替代 3 步主评价。", "",
             "## 数值差", "", "| 核验项 | 最大绝对差/残差 |", "|---|---:|" ]
    for name in ("response_head_coefficients", "response_head_column_scale", "response_saved_head_kkt",
                 "response_training_data_loss", "response_training_profiled_objective", "response_validation_loss",
                 "ridge_coefficients", "ridge_saved_coefficients_kkt", "ridge_validation_loss",
                 "native_test_predictions", "native_dense_path_predictions", "summary_Q"):
        item = result["numeric_checks"][name]
        value = item.get("max_absolute_gap", item.get("max_absolute_residual"))
        lines.append(f"| `{name}` | {value:.12g} |")
    lines.extend(["", "## 学习与预算", "",
                  f"32 个响应最终 checkpoint 中，{result['final_response_learning']['changed_nonlinear_checkpoints']} 个存在非零的已学习非线性参数变化；{result['final_response_learning']['selected_at_cap']} 个所选 epoch 达到 320 上限。完整分市场、设置和家族明细在 JSON 中保留。", "",
                  "这些检查验证保存的参数确实满足相应训练读出方程，并复现已发表在运行收据中的损失和预测。它们不重演 Adam 的完整参数轨迹，不证明非线性全局最优，也不把达到预算上限解释为已充分收敛。最佳 checkpoint 没有保存 trial 最后一次尝试参数，故该最终尝试向量的变化量只保留原收据，未声称独立重建。", "",
                  "所有输出仅含误差上界、计数、模型级参数变化和文件身份，不含逐起点数据或预测。信息设置及开发证据角色沿用冻结方案；本审计不改变科研结论。", "",
                  f"审计程序 SHA256：`{result['audit_source_sha256']}`。", "",
                  f"冻结方案 SHA256：`{result['plan_sha256']}`。", ""])
    if result["failures"]:
        lines.extend(["## 未通过项", "", "```json", json.dumps(result["failures"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, default=STUDY/"provenance/RUN_DATASETS.json")
    parser.add_argument("--plan", type=Path, default=STUDY/"audits/EXPERIMENT_PLAN_BEFORE_RUN.json")
    parser.add_argument("--output-json", type=Path, default=STUDY/"audits/FITTED_EQUATIONS_REPLAY.json")
    parser.add_argument("--output-md", type=Path, default=STUDY/"audits/FITTED_EQUATIONS_REPLAY_ZH.md")
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("Use fresh aggregate audit outputs")
    root = args.run_root.resolve()
    if root.is_relative_to(REPO):
        raise ValueError("Per-origin run inputs must be outside the public checkout")
    plan, plan_sha = read(args.plan), digest(args.plan)
    if digest(args.datasets) != plan["datasets_sha256"]:
        raise AssertionError("Dataset manifest differs from frozen plan")
    sources = {name: digest(STUDY/"source"/name) for name in plan["all_fitting_source_sha256"]}
    if sources != plan["all_fitting_source_sha256"]:
        raise AssertionError("A frozen fitting source changed")
    for rel, expected in plan["dense_path"]["source_sha256"].items():
        if digest(REPO/rel) != expected:
            raise AssertionError("A frozen dense-path dependency changed")
    checks = Checks()
    complete = assert_complete(root, plan, plan_sha, checks)
    specs = read(args.datasets)["datasets"]
    ledger, models, dataset_records = [], [], []
    for spec in specs:
        market = spec["market"]
        d = canonical_data(spec)
        dataset_records.append({"market": market, "sha256": spec["sha256"],
                                "row_boundaries": d["row_boundaries"],
                                "counts": {k: int(v.sum()) for k, v in d["masks"].items()}})
        for setting in SETTINGS:
            cache = ewt_cache(root/"ewt", market, setting, d, checks, ledger, root)
            x = {s: cache[f"x_{s}"] for s in ("train", "val", "final", "test")}
            labels = {s: cache[f"y_{s}"]/d["scale"] for s in ("train", "val", "final")}
            variance = d["variance"]/d["scale"]**2
            for family in FAMILIES:
                dest = root/"ewt"/f"{market}_{FOLD}"/setting/family
                models.append(response_job(dest, "ewt", market, setting, family, x, labels, variance, variance,
                                            cache["y_test"], cache["origins_test"], d["scale"], EVAL_HORIZONS,
                                            checks, ledger, root))
            models.append(ridge_job(root/"ewt"/f"{market}_{FOLD}"/setting/"ridge_price",
                                    market, setting, cache, checks, ledger, root))
        masks = d["masks"]
        x = {s: d["x"][masks[s]] for s in ("train", "val", "final", "test")}
        labels = {s: d["dense_y"][masks[s]]/d["scale"] for s in ("train", "final")}
        labels["val"] = d["y"][masks["val"]]/d["scale"]
        for family in FAMILIES:
            dest = root/"dense_path"/f"{market}_{FOLD}"/family
            models.append(response_job(dest, "dense", market, "dense_path", family, x, labels,
                                        d["dense_variance"]/d["scale"]**2, d["variance"]/d["scale"]**2,
                                        d["y"][masks["test"]], d["origins"][masks["test"]], d["scale"], DENSE_HORIZONS,
                                        checks, ledger, root))
    expected_counts = {"ewt_response_selection_checkpoints": 384, "ewt_response_final_checkpoints": 24,
                       "dense_response_selection_checkpoints": 128, "dense_response_final_checkpoints": 8,
                       "ewt_ridge_selection_checkpoints": 48, "ewt_ridge_final_checkpoints": 6}
    if any(checks.counts[k] != v for k, v in expected_counts.items()) or len(models) != 38:
        raise AssertionError("Independent replay did not cover all 598 fitted checkpoints")
    response_rows = [r for r in models if r["family"] != "ridge_price"]
    for block, source in (("ewt", complete["ewt"]), ("dense", complete["dense_path"])):
        rows = [r for r in response_rows if r["block"] == block]
        selected = sum(r["selection_positive_gradient_steps"] for r in rows)
        final = sum(r["refit_positive_gradient_steps"] for r in rows)
        if block == "ewt":
            if selected != source["selection_positive_gradient_steps"] or final != source["refit_positive_gradient_steps"]:
                raise AssertionError("EWT completion update counts differ")
        elif selected+final != source["positive_gradient_steps"]:
            raise AssertionError("Dense completion update counts differ")
    unchanged_sources = {name: digest(STUDY/"source"/name) for name in sources} == sources
    if not unchanged_sources or digest(args.plan) != plan_sha:
        raise AssertionError("Frozen source or plan changed during read-only replay")
    result = {"status": "PASS_INDEPENDENT_NUMPY_FITTED_EQUATIONS" if not checks.failures else "FAIL_NUMERICAL_REPLAY",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "audit_source_sha256": digest(__file__), "plan_sha256": plan_sha,
              "frozen_fitting_source_sha256": sources, "frozen_sources_and_plan_unchanged": True,
              "datasets": dataset_records, "counts": dict(checks.counts),
              "new_optimizer_training_runs": 0, "new_hyperparameter_selections": 0,
              "response_saved_head_equations_replayed": 544, "ridge_equations_replayed": 54,
              "final_prediction_sets_replayed": 38, "per_origin_output_arrays_written": 0,
              "implementation": {"basis": "independent explicit NumPy formula; no Torch model forward",
                                 "raw_features_labels_and_maturity": "independently reconstructed from identity-checked canonical files",
                                 "ewt_front_end": "identity-checked existing private preprocessing cache; transform itself not reimplemented",
                                 "head": "independent NumPy weighted standardized-column ridge solve including drift",
                                 "ridge": "independent scaler statistics and centered sum-objective ridge equation; no sklearn estimator",
                                 "selection": "fixed recorded trial table and epoch tolerance rule only; no new trial selected",
                                 "limitations": ["Does not replay Adam parameter trajectories or prove nonlinear global optimum",
                                                 "Does not independently reconstruct the last attempted nonlinear vector from a best-state-only checkpoint",
                                                 "Does not audit matured online readouts, which have a separate audit"]},
              "tolerances": {"absolute": ATOL, "relative": RTOL, "kkt_over_one_plus_rhs_max_abs": KKT_TOL},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "numeric_checks": checks.maxima, "failures": checks.failures,
              "final_response_learning": {"checkpoint_count": len(response_rows),
                                          "changed_nonlinear_checkpoints": sum(r["final_any_nonlinear_parameter_changed"] for r in response_rows),
                                          "nonzero_linear_readout_checkpoints": sum(r["final_linear_coefficients_nonzero"] for r in response_rows),
                                          "selected_at_cap": sum(r["selected_at_epoch_cap"] for r in response_rows)},
              "model_level_records": models, "consumed_file_identities": sorted(ledger, key=lambda x: x["path"])}
    write_new(args.output_json, result)
    write_new(args.output_md, markdown(result))
    print(json.dumps({"status": result["status"], "checkpoints": 598,
                      "final_prediction_sets": 38, "failures": len(checks.failures),
                      "output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False), flush=True)
    if checks.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
