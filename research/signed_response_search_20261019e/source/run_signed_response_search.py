"""Stage-e deterministic signed conditional response comparison.

Generic canonical date,price,volume_tons,turnover_yuan input. Price is retained
verbatim; no whole-series cleaning/decomposition. A run requires a frozen plan
and its expected digest. --preflight performs no market fit or forecast score.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from signed_response_models import (DTYPE, FAMILIES as RESPONSE_FAMILIES, HORIZONS,
                                   LOOKBACK, FitConfig, SignedResponse, fit_response,
                                   solve_readout)

SOURCE = Path(__file__).resolve().parent
STUDY = SOURCE.parent
OLD_SOURCE = STUDY.parent / "literature_aligned_model_search_20261019d/source"
OLD_SHA = {
    "run_model_search.py": "26347433fe6416298e4969da618cbba291999f3a66131172bc8f2e510f3633bf",
    "innovation_models.py": "cca83f45474c5110f180ae8b7763131f33596a664939c4ebd34c32db24b4dc1e",
    "run_liquidity_extension.py": "95665858a1945b00bba6d6233022c34c0a58993899ccadda8c895f64ffe69807",
    "liquidity_models.py": "91dd66c0b33b8d75100e37cc382f01be44021ae538f6cbbedddda7f9b9386e0b",
    "run_new_market_replication.py": "8c617a714bb0b47d117b213de9d299c3bd6a534c8211c28c8d039d454df12e56",
    "literature_diagnostic.py": "e9053bfd34bce9e60c67c0590245271ccecaaf0bcf009a1e715be9abe7ada2e5",
    "run_swewt_gru_comparison.py": "0b3395f547fad792c6880a9d25819c6e326ab48ef733240ad690e4833b63218d",
    "run_official_tianjin_extension.py": "2b47055caf840d7fe572365760a8a602edff844df7dc23dc2caf99717fc77e24",
}
# Source identity is verification only: no stage-d runner is imported or patched.
CLASSICAL = ("rw", "ridge_price", "ridge_interaction", "legacy_fixed_response")
ALL_FAMILIES = (*CLASSICAL, *RESPONSE_FAMILIES)
LRS = (.003, .01)
MEAN_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.)
PRICE_RIDGE_ALPHAS = (.01, .1, 1., 10., 100., 1000., 10000., 100000.)
FOLD = "fixed602020"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def source_hashes():
    return {p.name: digest(p) for p in [SOURCE/"signed_response_models.py", Path(__file__)]}


def check_old_sources():
    actual = {name: digest(OLD_SOURCE/name) for name in OLD_SHA}
    if actual != OLD_SHA:
        raise AssertionError({k: (OLD_SHA[k], actual[k]) for k in OLD_SHA if actual[k] != OLD_SHA[k]})
    return actual


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__,
            "precision": "float64 for all new response calculations; frozen legacy model float32",
            "torch_threads_per_worker": 1}


def read_datasets(path):
    payload = json.loads(Path(path).read_text())
    specs = payload["datasets"] if isinstance(payload, dict) else payload
    required = {"market", "file", "sha256", "role", "source_url", "price_definition", "source_processing", "license"}
    if not specs or len({s["market"] for s in specs}) != len(specs):
        raise ValueError("Provide unique nonempty market identities")
    for spec in specs:
        if not required.issubset(spec):
            raise ValueError(f"Missing dataset metadata: {sorted(required-set(spec))}")
        if spec["role"] not in ("development", "confirmation"):
            raise ValueError("Role must be development or confirmation")
        if any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_" for c in spec["market"]):
            raise ValueError("Use a filename-safe market identifier")
    return specs


def causal_tokens(price, volume, train_end):
    returns = np.r_[0., np.diff(price)]
    scale = max(float(np.sqrt(np.mean(returns[1:train_end]**2))), 1e-7)
    valid = np.isfinite(volume) & (volume >= 0)
    logs = np.zeros(len(volume), dtype=float)
    logs[valid] = np.log1p(volume[valid])
    context = np.zeros((len(price), 2), dtype=float)
    for t in range(len(price)):
        past_valid = valid[max(0, t-60):t]
        if valid[t] and past_valid.sum() >= 10:
            past = logs[max(0, t-60):t][past_valid]
            context[t] = [np.clip((logs[t]-np.median(past))/max(float(np.std(past)), .25), -5, 5), 1.]
    return np.column_stack((returns/scale, context)), scale, valid


def prepare(spec):
    path = Path(spec["file"])
    if digest(path) != spec["sha256"]:
        raise AssertionError(f"Input digest changed: {spec['market']}")
    frame = pd.read_csv(path)
    if list(frame) != ["date", "price", "volume_tons", "turnover_yuan"]:
        raise AssertionError("Canonical four-column schema required")
    if not frame.date.is_unique or not frame.date.is_monotonic_increasing:
        raise AssertionError("Canonical dates must be unique and increasing")
    dates = pd.to_datetime(frame.date).to_numpy(dtype="datetime64[D]")
    p, volume = frame.price.to_numpy(float), frame.volume_tons.to_numpy(float)
    turnover = frame.turnover_yuan.to_numpy(float)
    if not np.isfinite(p).all() or not (p > 0).all():
        raise AssertionError("Retained published prices must be finite and positive")
    if np.any(np.isfinite(volume) & (volume < 0)) or np.any(np.isfinite(turnover) & (turnover < 0)):
        raise AssertionError("Negative observed volume/turnover is not silently cleaned")
    n = len(p)
    a, b = n*3//5, n*3//5+n//5
    origins = np.arange(LOOKBACK, n-max(HORIZONS))
    masks = {"train": origins+max(HORIZONS) < a,
             "val": (origins >= a) & (origins+max(HORIZONS) < b),
             "final": origins+max(HORIZONS) < b,
             "test": origins >= b}
    if not all(mask.any() for mask in masks.values()):
        raise AssertionError("All four matured partitions must contain origins")
    if not (origins[masks["train"]][-1]+20 < origins[masks["val"]][0]
            and origins[masks["val"]][-1]+20 < origins[masks["test"]][0]):
        raise AssertionError("Label maturity separation failed")
    tokens, scale, valid = causal_tokens(p, volume, a)
    x = np.stack([tokens[t-LOOKBACK+1:t+1] for t in origins])
    y = np.column_stack([p[origins+h]-p[origins] for h in HORIZONS])
    variance = np.maximum(np.mean(y[masks["train"]]**2, axis=0), scale**2*1e-6)
    ratios = valid & (volume > 0) & np.isfinite(turnover)
    gap = float(np.max(np.abs(p[ratios]-turnover[ratios]/volume[ratios]))) if ratios.any() else None
    manifest = {"market": spec["market"], "role": spec["role"], "fold": FOLD,
                "target": spec["price_definition"], "row_boundaries": [0, a, b, n],
                "counts": {k: int(m.sum()) for k, m in masks.items()},
                "origin_ranges": {k: [int(origins[m][0]), int(origins[m][-1])] for k, m in masks.items()},
                "raw_date_range": [str(dates[0]), str(dates[-1])],
                "first_train_origin": str(dates[origins[masks['train']][0]]),
                "last_train_label": str(dates[origins[masks['train']][-1]+20]),
                "first_validation_origin": str(dates[origins[masks['val']][0]]),
                "last_validation_label": str(dates[origins[masks['val']][-1]+20]),
                "first_test_origin": str(dates[origins[masks['test']][0]]),
                "last_test_origin": str(dates[origins[masks['test']][-1]]),
                "last_test_label": str(dates[origins[masks['test']][-1]+20]),
                "training_rms": [scale], "train_target_second_moment": variance.tolist(),
                "sources": [{**spec, "path": str(path.resolve())}],
                "price_retained_without_turnover_division": True,
                "observed_price_turnover_ratio_max_gap_qc_only": gap,
                "maximum_record_gap_calendar_days": int(np.max(np.diff(dates).astype(int))),
                "input_columns": ["change_over_fixed_train_rms", "volume_surprise", "volume_valid"],
                "energy_inputs": False,
                "volume": {"valid_rows": int(valid.sum()), "usable_context_rows": int(tokens[:, 2].sum()),
                           "normalization": "current log1p volume minus prior60 target-record valid-log-volume median, divided by prior std floored .25, clipped[-5,5], >=10 valid past records",
                           "information": "origin price and volume observed after origin report; no future context; turnover not an input",
                           "missing": "0 surprise plus 0 valid mask; genuine observed zero volume is valid",
                           "model_usage": "only current origin context enters gates; historical price window64; current volume reference uses prior60 records"},
                "source_vintage": spec.get("source_vintage", "current public historical snapshot; daily publication vintages unverified")}
    return {"dates": dates, "price": p, "volume": volume, "origins": origins, "x": x, "y": y,
            "masks": masks, "scale": scale, "variance": variance, "manifest": manifest}


def tensor(array):
    return torch.as_tensor(array, dtype=DTYPE)


def normalized_labels(d):
    return tensor(d["y"]/d["scale"]), tensor(d["variance"]/d["scale"]**2)


def interaction_features(x):
    r = x[:, -20:, 0]
    z = np.column_stack((np.log1p(np.abs(x[:, -1, 0])), x[:, -1, 1]*x[:, -1, 2]))
    return np.concatenate((r, z, r*z[:, 0, None], r*z[:, 1, None]), axis=1)


def direct_basis(features):
    features = tensor(np.column_stack((np.ones(len(features)), features)))
    eye = torch.eye(len(HORIZONS), dtype=DTYPE)
    return (features[:, None, None, :]*eye[None, :, :, None]).reshape(len(features), len(HORIZONS), -1)


def legacy_prediction(x):
    path = OLD_SOURCE/"innovation_models.py"
    if digest(path) != OLD_SHA[path.name]:
        raise AssertionError("Legacy model changed")
    spec = importlib.util.spec_from_file_location("stage_d_frozen_innovation_for_e", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ShockAllocationForecaster(n_exog=3, horizon=20, variant="constant_allocation")
    packed = np.zeros((len(x), LOOKBACK, 4), dtype="float32")
    packed[:, :, 0] = x[:, :, 0]
    with torch.no_grad():
        return model(torch.from_numpy(packed)).numpy()[:, np.asarray(HORIZONS)-1]


def summarize(d, family, predicted):
    observed = d["y"][d["masks"]["test"]]
    rw = np.mean(observed**2, axis=0)
    if not (rw > 0).all():
        raise AssertionError("RW denominator is zero; report this case explicitly")
    mse = np.mean((predicted-observed)**2, axis=0)
    return {"market": d["manifest"]["market"], "role": d["manifest"]["role"],
            "fold": FOLD, "family": family, "n_origins": len(observed),
            "horizons": list(HORIZONS), "mse_by_horizon": mse.tolist(),
            "rw_mse_by_horizon": rw.tolist(), "mse_over_rw_by_horizon": (mse/rw).tolist(),
            "mean_mse_over_rw": float(np.mean(mse/rw)), "deterministic_prediction_count": 1,
            "no_seed_replication": True}


def run_response_job(spec, family, out):
    torch.set_num_threads(1)
    d = prepare(spec)
    dest = Path(out)/f"{spec['market']}_{FOLD}"/family
    m = d["masks"]
    x, (y, variance) = tensor(d["x"]), normalized_labels(d)
    trials = []
    rates = LRS if family != "fixed_response" else (None,)
    for alpha in MEAN_ALPHAS:
        for lr in rates:
            config = FitConfig(learning_rate=lr or LRS[0], alpha=alpha)
            _, receipt = fit_response(family, x[m["train"]], y[m["train"]], variance, config,
                                      x_val=x[m["val"]], y_val=y[m["val"]])
            name = f"selection_alpha{alpha:g}_lr{lr if lr is not None else 'none'}"
            write_json(dest/(name+".json"), receipt)
            torch.save({"family": family, "state_dict": receipt["state_dict"], "selected_epoch": receipt["selected_epoch"]}, dest/(name+".pt"))
            trials.append({"alpha": alpha, "learning_rate": lr,
                           "selected_epoch": receipt["selected_epoch"],
                           "validation_loss": receipt["validation_loss"], "receipt": name+".json",
                           "epochs_executed": receipt["epochs_executed"],
                           "positive_gradient_steps": receipt["positive_gradient_steps"]})
    choice = min(trials, key=lambda r: (r["validation_loss"], r["alpha"], r["learning_rate"] or 0, r["selected_epoch"]))
    config = FitConfig(learning_rate=choice["learning_rate"] or LRS[0], alpha=choice["alpha"])
    model, receipt = fit_response(family, x[m["final"]], y[m["final"]], variance, config,
                                  fixed_epochs=choice["selected_epoch"])
    write_json(dest/"SELECTION.json", {"selected": choice, "trials": trials,
                                      "alpha_unit": "mean weighted squared error + alpha*sum squared standardized coefficients",
                                      "epoch0": "global rates .5/.9, zero context gate, exact training ridge readout"})
    write_json(dest/"REFIT.json", receipt)
    torch.save({"family": family, "state_dict": model.state_dict()}, dest/"refit.pt")
    with torch.no_grad():
        pred = model(x[m["test"]]).numpy()*d["scale"]
        # These references are result-after diagnostics saved without selecting
        # another candidate: fixed rates/gates with the same selected alpha.
        init = SignedResponse(family)
        beta, _, _, _, colscale = solve_readout(init.basis(x[m["final"]]), y[m["final"]], variance, choice["alpha"])
        init.set_coefficients(beta, colscale)
        init_pred = init(x[m["test"]]).numpy()*d["scale"]
    np.savez_compressed(dest/"predictions.npz", predictions=pred[None],
                        fixed_dynamics_refit_readout_predictions=init_pred[None],
                        observed=d["y"][m["test"]], origins=d["origins"][m["test"]],
                        horizons=np.asarray(HORIZONS), deterministic=np.array([True]))
    result = summarize(d, family, pred)
    result["parameter_counts"] = model.parameter_counts()
    result["selected_epoch"] = choice["selected_epoch"]
    result["selection_positive_gradient_steps"] = sum(r["positive_gradient_steps"] for r in trials)
    result["refit_positive_gradient_steps"] = receipt["positive_gradient_steps"]
    write_json(dest/"SUMMARY.json", result)
    return result


def run_classical_job(spec, family, out):
    torch.set_num_threads(1)
    d = prepare(spec)
    dest = Path(out)/f"{spec['market']}_{FOLD}"/family
    m, x = d["masks"], d["x"]
    y, variance = d["y"], d["variance"]
    if family == "rw":
        pred, choice = np.zeros_like(y[m["test"]]), {"fitted_parameters": 0}
    elif family == "legacy_fixed_response":
        pred = legacy_prediction(x[m["test"]])*d["scale"]
        choice = {"fitted_parameters": 0, "legacy_source_sha256": OLD_SHA["innovation_models.py"],
                  "initial_rates": [.5, .9], "initial_transient_allocation": [.05, .05],
                  "energy_effect": "zero fixed coefficients; no energy input values read", "never_refitted": True}
    elif family == "ridge_price":
        features = x[:, -20:, 0]
        scaler = StandardScaler().fit(features[m["train"]])
        xx = scaler.transform(features)
        trials = []
        for alpha in PRICE_RIDGE_ALPHAS:
            model = Ridge(alpha=alpha).fit(xx[m["train"]], y[m["train"]])
            score = float(np.mean((model.predict(xx[m["val"]])-y[m["val"]])**2/variance))
            trials.append({"alpha": alpha, "validation_loss": score})
        best = min(trials, key=lambda r: (r["validation_loss"], r["alpha"]))
        final_scaler = StandardScaler().fit(features[m["final"]])
        final_x = final_scaler.transform(features)
        model = Ridge(alpha=best["alpha"]).fit(final_x[m["final"]], y[m["final"]])
        pred = model.predict(final_x[m["test"]])
        choice = {"selected": best, "trials": trials, "n_features": 20,
                  "alpha_unit": "sklearn Ridge unnormalized sum squared native error + alpha*sum coefficient squares; unpenalized fitted intercept",
                  "final_column_mean": final_scaler.mean_.tolist(), "final_column_scale": final_scaler.scale_.tolist(),
                  "coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist()}
    elif family == "ridge_interaction":
        features = interaction_features(x)
        if features.shape[1] != 62:
            raise AssertionError("Prespecified interaction basis must contain 62 features")
        basis = direct_basis(features)
        yn, vn = normalized_labels(d)
        trials = []
        for alpha in MEAN_ALPHAS:
            beta, _, _, _, _ = solve_readout(basis[m["train"]], yn[m["train"]], vn, alpha)
            score = float(torch.mean((basis[m["val"]]@beta-yn[m["val"]])**2/vn))
            trials.append({"alpha": alpha, "validation_loss": score})
        best = min(trials, key=lambda r: (r["validation_loss"], r["alpha"]))
        beta, _, _, residual, scales = solve_readout(basis[m["final"]], yn[m["final"]], vn, best["alpha"])
        pred = (basis[m["test"]]@beta).numpy()*d["scale"]
        choice = {"selected": best, "trials": trials, "n_features": 62,
                  "fitted_parameters": 63*3, "alpha_unit": "same mean weighted standardized-column ridge as response heads",
                  "feature_definition": "last20 r, current[log1p|r|, volume_surprise*valid], r*each of the two contexts",
                  "coefficients": beta.tolist(), "column_scale": scales.tolist(),
                  "normal_equation_max_residual": float(residual.abs().max())}
    else:
        raise ValueError(family)
    write_json(dest/"SELECTION.json", choice)
    np.savez_compressed(dest/"predictions.npz", predictions=pred[None], observed=y[m["test"]],
                        origins=d["origins"][m["test"]], horizons=np.asarray(HORIZONS),
                        deterministic=np.array([True]))
    result = summarize(d, family, pred)
    write_json(dest/"SUMMARY.json", result)
    return result


def preflight(specs):
    torch.set_num_threads(1)
    old = check_old_sources()
    records = []
    for spec in specs:
        d = prepare(spec)
        x = tensor(d["x"][:3])
        checks = {}
        for family in RESPONSE_FAMILIES:
            model = SignedResponse(family)
            with torch.no_grad():
                prediction = model(x)
                basis = model.basis(x)
            if prediction.shape != (3, 3) or not bool(torch.isfinite(basis).all()):
                raise AssertionError(f"Nonfinite/invalid forward: {family}")
            checks[family] = {"prediction_shape": list(prediction.shape),
                              "basis_shape": list(basis.shape), **model.parameter_counts()}
        legacy = legacy_prediction(d["x"][:3])
        assert legacy.shape == (3, 3) and np.isfinite(legacy).all()
        a, b = d["manifest"]["row_boundaries"][1:3]
        perturbed_price, perturbed_volume = d["price"].copy(), d["volume"].copy()
        perturbed_price[b+1:] += 12345.
        perturbed_volume[b+1:] *= 7.
        before, scale_before, _ = causal_tokens(d["price"], d["volume"], a)
        after, scale_after, _ = causal_tokens(perturbed_price, perturbed_volume, a)
        assert scale_before == scale_after and np.array_equal(before[:b+1], after[:b+1])
        assert np.array_equal(SignedResponse("signed_context_response").basis(x).detach().numpy(),
                              SignedResponse("amplitude_context").basis(x).detach().numpy())
        records.append({"input_manifest": d["manifest"], "forward_checks": checks,
                        "legacy_shape": list(legacy.shape), "future_perturbation_prefix_gap": 0.,
                        "full_amplitude_zero_gate_basis_equal": True,
                        "ridge_interaction_feature_count": interaction_features(d["x"][:3]).shape[1]})
    return {"status": "PASS_NO_MARKET_FIT_NO_SCORE", "market_fit_count": 0,
            "validation_or_test_model_scores": 0, "source_sha256": source_hashes(),
            "old_frozen_source_sha256": old, "runtime": runtime(), "datasets": records}


def freeze_run(specs, out, plan, expected_plan_sha):
    if digest(plan) != expected_plan_sha:
        raise AssertionError("Expected frozen plan digest does not match")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("New experiment requires an empty output directory")
    old = check_old_sources()
    protocol = {"stage": "signed_response_search_20261019e", "created_unix": time.time(),
                "frozen_analysis_plan": str(plan.resolve()), "frozen_analysis_plan_sha256": expected_plan_sha,
                "source_sha256": source_hashes(), "old_frozen_source_sha256": old,
                "datasets": specs, "families": list(ALL_FAMILIES),
                "primary_candidate": "signed_context_response", "horizons": list(HORIZONS),
                "lookback": LOOKBACK, "learning_rates": list(LRS),
                "mean_weighted_ridge_alphas": list(MEAN_ALPHAS), "price_ridge_sum_alphas": list(PRICE_RIDGE_ALPHAS),
                "fit_config_defaults": FitConfig().__dict__,
                "deterministic": True, "seed_repeats": 0,
                "selection": "training-only profiled weighted ridge plus nonlinear Adam; validation selects alpha/lr/epoch; ties use alpha then lr then epoch",
                "refit": "fresh initialization; all H20-mature pretest labels; exactly selected nonlinear epoch count; recompute training head scales and ridge beta each update",
                "epoch0": "fitted signed linear readout on fixed initial rates, not a completely untrained fixed prediction",
                "past_information": "64 historical price changes and current activity/volume context; volume reference prior60 target records; no energy or future context",
                "global_normalization": "fixed initial-train price RMS and H-specific training target second moments throughout selection/refit/test",
                "dynamic_interpretation": "origin-conditioned discriminative forecast, frozen future rates within each issued horizon; no generative or rolling tower-property claim",
                "initialization_diagnostic": "same chosen alpha and initial nonlinear rates/gates, readout exactly refitted on final training labels; saved without selection or extra p-values",
                "planned_selection_fits_per_market": 5*2*8+8+8+8,
                "planned_final_fit_constructions_per_market": 8,
                "additional_fixed_dynamics_linear_readout_diagnostic_solves_per_market": 6,
                "linear_solve_count_definition": "Every response epoch including epoch0 solves one readout;8 ridge_interaction selection+1 refit,8 ridge_price selection+1 refit,plus6 fixed-dynamics diagnostic solves. Report actual readout solves separately from nonlinear optimizer steps.",
                "fixed_formula_arms_per_market": 2,
                "runtime": runtime()}
    write_json(out/"PROTOCOL_BEFORE_RUN.json", protocol)
    for spec in specs:
        d = prepare(spec)
        write_json(out/f"{spec['market']}_{FOLD}"/"INPUT_MANIFEST.json", d["manifest"])
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-plan", type=Path)
    parser.add_argument("--freeze-plan-sha")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    specs = read_datasets(args.datasets)
    if args.preflight:
        if args.preflight_report is None:
            parser.error("--preflight requires an explicit --preflight-report path")
        if args.preflight_report.exists():
            raise FileExistsError("Do not overwrite a previous preflight receipt")
        receipt = preflight(specs)
        write_json(args.preflight_report, receipt)
        print(json.dumps({"status": receipt["status"], "report": str(args.preflight_report), "source_sha256": receipt["source_sha256"]}), flush=True)
        return
    if args.out is None or args.freeze_plan is None or not args.freeze_plan_sha:
        parser.error("Run requires --out, --freeze-plan, --freeze-plan-sha")
    protocol = freeze_run(specs, args.out, args.freeze_plan, args.freeze_plan_sha)
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {}
        for spec in specs:
            for family in ALL_FAMILIES:
                worker = run_response_job if family in RESPONSE_FAMILIES else run_classical_job
                tasks[pool.submit(worker, spec, family, str(args.out))] = f"{spec['market']}/{family}"
        for fut in as_completed(tasks):
            result = fut.result()
            summaries.append(result)
            write_json(args.out/"PROGRESS_SUMMARIES.json", summaries)
            print(json.dumps({"completed": tasks[fut], "mean_mse_over_rw": result["mean_mse_over_rw"]}), flush=True)
    write_json(args.out/"ALL_SUMMARIES.json", summaries)
    new_hashes, old_hashes = source_hashes(), check_old_sources()
    unchanged = new_hashes == protocol["source_sha256"] and old_hashes == protocol["old_frozen_source_sha256"]
    plan_unchanged = digest(args.freeze_plan) == args.freeze_plan_sha
    write_json(args.out/"EXECUTION_COMPLETE.json", {"status": "completed" if unchanged and plan_unchanged else "IDENTITY_FAILURE",
               "summary_rows": len(summaries), "source_hashes_unchanged": unchanged,
               "plan_hash_unchanged": plan_unchanged, "deterministic": True})
    if not unchanged or not plan_unchanged:
        raise AssertionError("Frozen execution identity changed")


if __name__ == "__main__":
    main()
