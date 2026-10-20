"""Fixed stage-f EWT input adaptations of the existing signed response model.

No run occurs without --plan and its expected SHA. --preflight checks source,
data, transforms and forward equivalence only: no fit, no market loss. Output
and preprocessing caches must be outside the checkout. Raw controls reuse the
completed stage-e320 response results and the original exact price Ridge/RW.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from signed_response_enhancements import (
    E_SOURCE, DTYPE, HORIZONS, LOOKBACK, FitConfig, SignedResponse,
    StateInputSignedResponse, RESPONSE_FAMILIES, SETTINGS, NEW_SETTINGS,
    EWT_OPTIONS, array_digest, digest, ewt_dependency,
    ewt_analysis, fit_response_with_factory,
    transformed_inputs, window_changes)
import run_signed_response_search as original

SOURCE = Path(__file__).resolve().parent
STUDY = SOURCE.parent
CHECKOUT = STUDY.parents[1]
E_STUDY = E_SOURCE.parent
LRS = original.LRS
ALPHAS = original.MEAN_ALPHAS
PRICE_ALPHAS = original.PRICE_RIDGE_ALPHAS
FOLD = original.FOLD
MAX_EPOCHS = 320
E_SOURCE_SHA = {
    "signed_response_models.py": "5ceb7f04c567be32f1a399cbf01dafdfc6894d3840a158b6faae4153f09eb2f8",
    "run_signed_response_search.py": "905da0ca33607799b1ca63d9efe6710df173ec33c5c44e137071e024906d990c",
    "run_neural_references.py": "489249eb417e54611123da2b1508f946f416c7f5813124e2ddd2eedfe8b63b11",
    "run_budget_sensitivity.py": "c353e5d386bf96ed8a9a4d048cffe152de08d9303a0e2551c2df755e51fb606b",
}
DATA_SHA = {
    "TIANJIN_OFFICIAL_MEAN": "85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a",
    "CEA_OFFICIAL_MEAN": "f9f0e09b7500b2bdb4ec8b16c61d5d5fa2b64de63a24c0cf7bd1d269aa2be75d",
}
EWT_SHA = "35cb4532ae35a41867ee1411cb61a33165f7eebab3ebf70a58b47c74c6e6a289"
EXPORT_RECEIPT = E_STUDY/"provenance/RESULT_EXPORT_RECEIPT_WITH_BUDGET.json"
EXPORT_RECEIPT_SHA = "2e715b5c35deb917f60c89bd38e85300c92ae9968b3139f9372e412066f0bc4b"


def write(path, value):
    original.write_json(path, value)


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {name: digest(SOURCE/name) for name in
            ("signed_response_enhancements.py", "run_primary_model_tricks.py")}


def verify_dependencies():
    e_actual = {name: digest(E_SOURCE/name) for name in E_SOURCE_SHA}
    if e_actual != E_SOURCE_SHA:
        raise AssertionError("An immutable stage-e dependency changed")
    old = original.check_old_sources()
    ewt = ewt_dependency()
    if ewt["version"] != "0.2" or ewt["source_sha256"] != EWT_SHA:
        raise AssertionError("The fixed ewtpy implementation changed")
    if digest(EXPORT_RECEIPT) != EXPORT_RECEIPT_SHA:
        raise AssertionError("Raw-control export identity changed")
    return {"stage_e": e_actual, "stage_d": old, "ewtpy": ewt,
            "raw_export_receipt_sha256": EXPORT_RECEIPT_SHA}


def specs_from(path):
    specs = original.read_datasets(path)
    if {s["market"] for s in specs} != set(DATA_SHA):
        raise AssertionError("Both specified markets, and no additional market, are required")
    result = []
    for source in specs:
        if source["sha256"] != DATA_SHA[source["market"]] or digest(source["file"]) != source["sha256"]:
            raise AssertionError("A canonical dataset changed")
        spec = copy.deepcopy(source)
        spec["historical_input_role"] = source["role"]
        spec["role"] = "development"
        spec["stage_f_exposure"] = "Both tails previously scored in stage e; all new settings are development"
        result.append(spec)
    return result


def scientific_configuration():
    return {"settings": list(SETTINGS), "new_fitted_settings": list(NEW_SETTINGS),
            "response_families": list(RESPONSE_FAMILIES), "matched_classical": ["ridge_price"],
            "raw_reuse": "stage-e320 four response families; original stage-e price Ridge and RW",
            "primary_family": "signed_context_response", "horizons": list(HORIZONS),
            "history_changes": LOOKBACK, "price_support": LOOKBACK+1,
            "ewt_options": EWT_OPTIONS, "analysis_bands": [0, 1, 2],
            "front_end": "raw changes plus differences of all3 EWT analysis coefficient channels; no reconstruction or band deletion",
            "state_bank": "four channels each with the same two history rates and the same future gate",
            "fitted_scalar_counts": {"signed_context_response": 15, "static_response": 11,
                                     "amplitude_context": 15, "signed_shock_only": 13},
            "shared_readout": "one drift plus8 signed response coefficients; 9 linear coefficients",
            "gate_inputs": "unchanged raw current log1p_abs_normalized_change and volume_surprise_times_valid",
            "prediction_anchor": "observed unfiltered origin price; no denoised anchor offset",
            "training_targets": "unfiltered official H1/H5/H20 cumulative price changes",
            "normalization": "unchanged initial-train RMS and target second moments",
            "split": "original60/20/20 masks; commonH20 maturity separation; no dropped origins",
            "joint_selection_price_end": "a-1", "joint_final_price_end": "b-1",
            "joint_validation_and_test": "per-origin65-price window",
            "full_series_scope": "allN prices once; separately labeled retrospective benchmark",
            "learning_rates": list(LRS), "mean_ridge_alphas": list(ALPHAS),
            "price_ridge_sum_alphas": list(PRICE_ALPHAS),
            "fit_config": asdict(FitConfig(maximum_epochs=MAX_EPOCHS)),
            "selection_rule": "minimum validation loss; alpha, learning_rate, selected_epoch ties; original epoch tolerance",
            "price_ridge_input": "last20 raw changes plus last20 changes from each of3 EWT analysis channels;80 features; training-only StandardScaler; free3h heads with intercept",
            "new_response_selection_fits": 384, "new_response_refits": 24,
            "new_ridge_selection_fits": 48, "new_ridge_refits": 6,
            "new_fit_tasks": 462, "new_prediction_sets": 30,
            "reused_prediction_sets": 12, "random_seed_repetitions": 0,
            "initial_dynamics_diagnostic_head_solves": 24,
            "data_sha256": DATA_SHA, "all_markets_role": "development"}


def require_private(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(CHECKOUT):
        raise ValueError("Raw inputs, caches, per-origin outputs and binary states must stay outside the checkout")
    return path


def prepare(spec, setting, *, base=None, window_cache=None):
    """Public adapter API: original d plus x_by_subset and transform_manifest."""
    d = original.prepare(spec) if base is None else base
    arrays, manifest = transformed_inputs(d, setting, window_cache=window_cache)
    return {**d, "x_by_subset": arrays, "transform_manifest": manifest, "setting": setting}


def make_caches(specs, out):
    caches = {}
    for spec in specs:
        d = original.prepare(spec)
        market = spec["market"]
        write(out/f"{market}_{FOLD}"/"INPUT_MANIFEST.json", d["manifest"])
        windows = np.stack([window_changes(d["price"], int(t)) for t in d["origins"]])
        for setting in NEW_SETTINGS:
            prepared = prepare(spec, setting, base=d, window_cache=windows)
            cache = out/"_private_preprocess"/f"{market}_{setting}.npz"
            cache.parent.mkdir(parents=True, exist_ok=True)
            payload = {"scale": np.asarray(d["scale"]), "variance": d["variance"]}
            for subset, x in prepared["x_by_subset"].items():
                payload[f"x_{subset}"] = x
                payload[f"y_{subset}"] = d["y"][d["masks"][subset]]
                payload[f"origins_{subset}"] = d["origins"][d["masks"][subset]]
            np.savez_compressed(cache, **payload)
            record = {**prepared["transform_manifest"], "market": market, "role": "development",
                      "cache_path": str(cache), "cache_sha256": digest(cache),
                      "canonical_sha256": spec["sha256"]}
            write(out/f"{market}_{FOLD}"/setting/"PREPROCESS_MANIFEST.json", record)
            caches[(market, setting)] = record
    return caches


def load_cache(record):
    if digest(record["cache_path"]) != record["cache_sha256"]:
        raise AssertionError("Frozen preprocessing cache changed")
    with np.load(record["cache_path"], allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    for subset in ("train", "val", "final", "test"):
        if array_digest(d[f"x_{subset}"]) != record["subsets"][subset]["input_sha256"]:
            raise AssertionError("Subset input digest mismatch")
        if array_digest(d[f"y_{subset}"]) != record["subsets"][subset]["native_target_sha256"]:
            raise AssertionError("Unfiltered target digest mismatch")
    return d


def summary(market, setting, family, prediction, y):
    mse = np.mean((prediction-y)**2, axis=0)
    rw = np.mean(y*y, axis=0)
    if not np.all(rw > 0):
        raise FloatingPointError("Zero RW reference denominator")
    return {"market": market, "role": "development", "setting": setting,
            "information_scope": "retrospective_full_sequence" if setting == "full_series_joint" else "origin_past_price_available_current_snapshot",
            "family": family, "fold": FOLD, "n_origins": len(y), "horizons": list(HORIZONS),
            "mse_by_horizon": mse.tolist(), "rmse_by_horizon": np.sqrt(mse).tolist(),
            "mae_by_horizon": np.mean(np.abs(prediction-y), axis=0).tolist(),
            "rw_mse_by_horizon": rw.tolist(), "mse_over_rw_by_horizon": (mse/rw).tolist(),
            "mean_mse_over_rw": float(np.mean(mse/rw)), "deterministic_prediction_count": 1,
            "no_seed_replication": True, "new_fit_tasks": 17 if family in RESPONSE_FAMILIES else 9}


def run_response_job(record, family, out):
    torch.set_num_threads(1)
    d = load_cache(record)
    market, setting = record["market"], record["setting"]
    dest = Path(out)/f"{market}_{FOLD}"/setting/family
    x = {s: torch.as_tensor(d[f"x_{s}"], dtype=DTYPE) for s in ("train", "val", "final", "test")}
    scale = float(d["scale"])
    y = {s: torch.as_tensor(d[f"y_{s}"]/scale, dtype=DTYPE) for s in ("train", "val", "final")}
    variance = torch.as_tensor(d["variance"]/scale**2, dtype=DTYPE)
    trials = []
    for alpha in ALPHAS:
        for lr in LRS:
            config = FitConfig(learning_rate=lr, alpha=alpha, maximum_epochs=MAX_EPOCHS)
            _, receipt = fit_response_with_factory(family, x["train"], y["train"], variance, config,
                                                    x_val=x["val"], y_val=y["val"])
            receipt.update({"setting": setting, "market": market, "role": "development",
                            "preprocess_cache_sha256": record["cache_sha256"]})
            name = f"selection_alpha{alpha:g}_lr{lr}"
            write(dest/(name+".json"), receipt)
            torch.save({"family": family, "state_dict": receipt["state_dict"],
                        "selected_epoch": receipt["selected_epoch"]}, dest/(name+".pt"))
            trials.append({"alpha": alpha, "learning_rate": lr, "selected_epoch": receipt["selected_epoch"],
                           "validation_loss": receipt["validation_loss"], "receipt": name+".json",
                           "epochs_executed": receipt["epochs_executed"],
                           "positive_gradient_steps": receipt["positive_gradient_steps"]})
    choice = min(trials, key=lambda r: (r["validation_loss"], r["alpha"], r["learning_rate"], r["selected_epoch"]))
    config = FitConfig(learning_rate=choice["learning_rate"], alpha=choice["alpha"], maximum_epochs=MAX_EPOCHS)
    model, receipt = fit_response_with_factory(family, x["final"], y["final"], variance, config,
                                                fixed_epochs=choice["selected_epoch"])
    receipt.update({"setting": setting, "market": market, "role": "development",
                    "preprocess_cache_sha256": record["cache_sha256"]})
    write(dest/"SELECTION.json", {"selected": choice, "trials": trials,
          "alpha_unit": "mean weighted original-price risk plus standardized-column ridge penalty",
          "train_and_final_inputs_separate": True})
    write(dest/"REFIT.json", receipt)
    torch.save({"family": family, "state_dict": model.state_dict()}, dest/"refit.pt")
    with torch.no_grad():
        prediction = model(x["test"]).numpy()*scale
        init = StateInputSignedResponse(family)
        beta, _, _, _, colscale = original.solve_readout(init.basis(x["final"]), y["final"], variance, choice["alpha"])
        init.set_coefficients(beta, colscale)
        init_prediction = init(x["test"]).numpy()*scale
    np.savez_compressed(dest/"predictions.npz", predictions=prediction[None],
                        fixed_dynamics_refit_readout_predictions=init_prediction[None],
                        observed=d["y_test"], origins=d["origins_test"],
                        horizons=np.asarray(HORIZONS), deterministic=np.asarray([True]))
    result = summary(market, setting, family, prediction, d["y_test"])
    result.update({"parameter_counts": model.parameter_counts(), "selected_epoch": choice["selected_epoch"],
                   "selection_positive_gradient_steps": sum(r["positive_gradient_steps"] for r in trials),
                   "refit_positive_gradient_steps": receipt["positive_gradient_steps"],
                   "selected_alpha": choice["alpha"], "selected_learning_rate": choice["learning_rate"]})
    write(dest/"SUMMARY.json", result)
    return result


def run_ridge_job(record, out):
    torch.set_num_threads(1)
    d = load_cache(record)
    market, setting = record["market"], record["setting"]
    dest = Path(out)/f"{market}_{FOLD}"/setting/"ridge_price"
    features = {s: d[f"x_{s}"][:, -20:, [0, 3, 4, 5]].transpose(0, 2, 1).reshape(len(d[f"x_{s}"]), 80)
                for s in ("train", "val", "final", "test")}
    scaler = StandardScaler().fit(features["train"])
    train = scaler.transform(features["train"])
    val = scaler.transform(features["val"])
    trials = []
    for alpha in PRICE_ALPHAS:
        model = Ridge(alpha=alpha).fit(train, d["y_train"])
        loss = float(np.mean((model.predict(val)-d["y_val"])**2/d["variance"]))
        name = f"selection_alpha{alpha:g}"
        receipt = {"alpha": alpha, "validation_loss": loss,
                   "coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
                   "column_mean": scaler.mean_.tolist(), "column_scale": scaler.scale_.tolist(),
                   "setting": setting, "family": "ridge_price", "market": market, "role": "development"}
        write(dest/(name+".json"), receipt)
        trials.append({"alpha": alpha, "validation_loss": loss, "receipt": name+".json"})
    choice = min(trials, key=lambda r: (r["validation_loss"], r["alpha"]))
    final_scaler = StandardScaler().fit(features["final"])
    model = Ridge(alpha=choice["alpha"]).fit(final_scaler.transform(features["final"]), d["y_final"])
    prediction = model.predict(final_scaler.transform(features["test"]))
    write(dest/"SELECTION.json", {"selected": choice, "trials": trials,
          "n_features": 80, "alpha_unit": "original sklearn sum native squared error; unpenalized intercept"})
    write(dest/"REFIT.json", {"alpha": choice["alpha"], "family": "ridge_price", "setting": setting,
          "coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
          "column_mean": final_scaler.mean_.tolist(), "column_scale": final_scaler.scale_.tolist(),
          "fitted_parameters": 243, "new_fit_tasks": 1})
    np.savez_compressed(dest/"predictions.npz", predictions=prediction[None], observed=d["y_test"],
                        origins=d["origins_test"], horizons=np.asarray(HORIZONS), deterministic=np.asarray([True]))
    result = summary(market, setting, "ridge_price", prediction, d["y_test"])
    result.update({"fitted_parameters": 243, "selected_alpha": choice["alpha"],
                   "selection_positive_gradient_steps": 0, "refit_positive_gradient_steps": 0})
    write(dest/"SUMMARY.json", result)
    return result


def raw_reference_records(specs, response_run, classical_run):
    """Verify exact earlier evidence; no refit, new selection or rescoring."""
    export = read(EXPORT_RECEIPT)
    records = []
    for group, root, families in (("budget320", Path(response_run), RESPONSE_FAMILIES),
                                   ("core", Path(classical_run), ("ridge_price", "rw"))):
        public = {r["path"].split(f"results/{group}/", 1)[1]: r["sha256"]
                  for r in export["runs"][group]["files"]}
        private = {r["filename"]: r["sha256"] for r in export["runs"][group]["private_prediction_files"]}
        for spec in specs:
            for family in families:
                rel = Path(f"{spec['market']}_{FOLD}")/family
                summary_path = root/rel/"SUMMARY.json"
                prediction_path = root/rel/"predictions.npz"
                if digest(summary_path) != public[str(rel/"SUMMARY.json")]:
                    raise AssertionError("Raw reused summary identity changed")
                if digest(prediction_path) != private[str(rel/"predictions.npz")]:
                    raise AssertionError("Raw reused prediction identity changed")
                records.append({"market": spec["market"], "family": family, "source_group": group,
                                "summary_path": str(summary_path), "summary_sha256": digest(summary_path),
                                "prediction_path": str(prediction_path), "prediction_sha256": digest(prediction_path)})
    return records


def copy_raw_references(records, out):
    rows = []
    for record in records:
        dest = out/f"{record['market']}_{FOLD}"/"raw"/record["family"]
        dest.mkdir(parents=True, exist_ok=True)
        row = read(record["summary_path"])
        row.update({"historical_execution_role": row.get("role"), "role": "development", "setting": "raw",
                    "reused_existing_run": True, "new_fit_tasks": 0,
                    "information_scope": "origin_past_price_available_current_snapshot",
                    "new_selection_positive_gradient_steps": 0, "new_refit_positive_gradient_steps": 0})
        write(dest/"SUMMARY.json", row)
        write(dest/"REUSED_BASELINE.json", record)
        shutil.copyfile(record["prediction_path"], dest/"predictions.npz")
        rows.append(row)
    return rows


def preflight(specs, raw_records):
    torch.set_num_threads(1)
    dependencies = verify_dependencies()
    analysis_checks = []
    for length in (65, 97):
        t = np.arange(length)
        p = 30+.02*t+np.sin(.2*t)+.3*np.cos(2.6*t)
        coefficients = ewt_analysis(p)
        delta = np.diff(coefficients, axis=0)
        if coefficients.shape != (length, 3) or delta.shape != (length-1, 3):
            raise AssertionError("All3 analysis channels must retain their time index")
        analysis_checks.append({"length": length, "analysis_shape": list(coefficients.shape),
                                "difference_shape": list(delta.shape), "all3_bands_used": True,
                                "inverse_transform_used": False})
    rng = np.random.default_rng(71019)
    raw = torch.as_tensor(rng.normal(size=(7, LOOKBACK, 3)), dtype=DTYPE)
    raw[..., 2] = (raw[..., 2] > 0).to(DTYPE)
    packed = torch.cat((raw, torch.zeros_like(raw)), dim=-1)
    parity = {}
    for family in RESPONSE_FAMILIES:
        old, new = SignedResponse(family), StateInputSignedResponse(family)
        with torch.no_grad():
            for parameter in old.parameters():
                parameter.add_(.17)
            old.coefficients.copy_(torch.as_tensor([.03, -.4, .2], dtype=DTYPE))
            new.rate_raw.copy_(old.rate_raw)
            new.gate.copy_(old.gate)
            new.coefficients[:3].copy_(old.coefficients)
            gap = float(torch.max(torch.abs(old(raw)-new(packed))))
            basis_gap = float(torch.max(torch.abs(old.basis(raw)-new.basis(packed)[..., :3])))
        expected = {"signed_context_response": 15, "static_response": 11,
                    "amplitude_context": 15, "signed_shock_only": 13}
        if gap > 1e-12 or basis_gap > 1e-12 or new.parameter_counts()["total_fitted_scalars"] != expected[family]:
            raise AssertionError("The original predictor is not nested in the expanded bank")
        altered = packed.clone()
        altered[..., 3] += .8
        _, old_context, _ = new.states_and_context(packed)
        _, new_context, _ = new.states_and_context(altered)
        if not torch.equal(old_context, new_context):
            raise AssertionError("Filtered state inputs changed the raw gate context")
        parity[family] = {"prediction_max_gap": gap, "basis_max_gap": basis_gap,
                          "parameter_counts": new.parameter_counts(), "raw_gate_context_unchanged": True}
    # Nonzero added channels check the new column order independently of the
    # zero-padded nesting check above. This is a synthetic forward calculation,
    # without target labels, a head solve or an optimizer update.
    all_channels = np.concatenate((raw.numpy(), rng.normal(size=(7, LOOKBACK, 3))), axis=-1)
    manual_context = np.column_stack((np.log1p(np.abs(all_channels[:, -1, 0])),
                                      all_channels[:, -1, 1]*all_channels[:, -1, 2]))

    def manual_rates(value):
        first = .98/(1.+np.exp(-value[..., 0]))
        second = first+(.98-first)/(1.+np.exp(-value[..., 1]))
        return np.stack((first, second), axis=-1)

    channel_formula = {}
    for family in RESPONSE_FAMILIES:
        model = StateInputSignedResponse(family)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.17)
            actual = model.basis(torch.as_tensor(all_channels, dtype=DTYPE)).numpy()
        raw_rates = model.rate_raw.detach().numpy()
        rho = manual_rates(raw_rates)
        gate_input = manual_context[:, :1] if family == "signed_shock_only" else manual_context
        shift = 2*np.tanh(gate_input @ model.gate.detach().numpy().T)
        future = (manual_rates(raw_rates[None]+shift)
                  if family in ("signed_context_response", "signed_shock_only")
                  else np.broadcast_to(rho, (len(all_channels), 2)))
        amplitude = np.exp(shift) if family == "amplitude_context" else np.ones_like(shift)
        expected_basis = np.zeros((len(all_channels), len(HORIZONS), 9))
        for hi, horizon in enumerate(HORIZONS):
            expected_basis[:, hi, 0] = horizon
            for ci, column in enumerate((0, 3, 4, 5)):
                for k in range(2):
                    state = sum((1-rho[k])*rho[k]**lag*all_channels[:, -1-lag, column]
                                for lag in range(LOOKBACK))
                    duration = sum(future[:, k]**j for j in range(1, horizon+1))
                    expected_basis[:, hi, 1+2*ci+k] = state*duration*amplitude[:, k]
        gap = float(np.max(np.abs(actual-expected_basis)))
        if gap > 1e-12:
            raise AssertionError("Raw plus all3 analysis-channel formula or packing changed")
        channel_formula[family] = {"independent_numpy_basis_max_gap": gap,
                                   "basis_columns": "drift, raw_rho0, raw_rho1, band0_rho0, band0_rho1, band1_rho0, band1_rho1, band2_rho0, band2_rho1"}
    ridge_packed = all_channels[:, -20:, [0, 3, 4, 5]].transpose(0, 2, 1).reshape(len(all_channels), 80)
    ridge_manual = np.concatenate([all_channels[:, -20:, column] for column in (0, 3, 4, 5)], axis=1)
    if not np.array_equal(ridge_packed, ridge_manual):
        raise AssertionError("Matched Ridge feature packing changed")
    markets = []
    for spec in specs:
        d = original.prepare(spec)
        _, a, b, n = d["manifest"]["row_boundaries"]
        windows = np.stack([window_changes(d["price"], int(t)) for t in d["origins"]])
        transforms = {}
        for setting in SETTINGS:
            prepared = prepare(spec, setting, base=d, window_cache=windows)
            transforms[setting] = prepared["transform_manifest"]
            for subset, x in prepared["x_by_subset"].items():
                with torch.no_grad():
                    pred = StateInputSignedResponse()(torch.as_tensor(x[:2], dtype=DTYPE))
                if pred.shape != (2, 3) or not torch.isfinite(pred).all():
                    raise AssertionError("Model forward shape failed")
                if setting == "raw" and not np.array_equal(x[..., 3:], np.zeros_like(x[..., 3:])):
                    raise AssertionError("Raw reference padding is not exactly zero")
        t = int(d["origins"][d["masks"]["val"]][0])
        altered = d["price"].copy()
        altered[t+1:] += 17.+np.arange(len(altered)-t-1)*.3
        gap = float(np.max(np.abs(window_changes(d["price"], t)-window_changes(altered, t))))
        if gap != 0:
            raise AssertionError("A future price entered window preprocessing")
        # Verify joint segments are explicitly bounded before transform calls.
        altered_a, altered_b = d["price"].copy(), d["price"].copy()
        altered_a[a:] += 100.
        altered_b[b:] -= 50.
        if not np.array_equal(ewt_analysis(d["price"][:a]), ewt_analysis(altered_a[:a])):
            raise AssertionError("Selection joint prefix was not isolated")
        if not np.array_equal(ewt_analysis(d["price"][:b]), ewt_analysis(altered_b[:b])):
            raise AssertionError("Final joint prefix was not isolated")
        markets.append({"market": spec["market"], "canonical_sha256": spec["sha256"],
                        "row_boundaries": [0, a, b, n], "counts": d["manifest"]["counts"],
                        "future_window_perturbation_max_gap": gap, "transforms": transforms})
    return {"status": "PASS_PREFLIGHT_NO_FIT_NO_MARKET_SCORE", "model_fits": 0,
            "linear_readout_solves": 0, "optimizer_updates": 0, "validation_or_test_forecast_scores": 0,
            "source_sha256": source_hashes(), "dependency_sha256": dependencies,
            "scientific_configuration": scientific_configuration(),
            "synthetic_analysis_shapes": analysis_checks,
            "formal_preprocessing_uses_analysis_features_only": True,
            "original_raw_model_nested_forward_parity": parity,
            "nonzero_all_channel_formula_checks": channel_formula,
            "matched_ridge80_column_packing_exact": True, "markets": markets,
            "verified_reused_raw_records": raw_records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--raw-response-run", type=Path, default=Path("/tmp/carbon-signed-response-runs-20261019e/budget320"))
    parser.add_argument("--raw-classical-run", type=Path, default=Path("/tmp/carbon-signed-response-runs-20261019e/core"))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    specs = specs_from(args.datasets)
    dependencies = verify_dependencies()
    raw_records = raw_reference_records(specs, args.raw_response_run, args.raw_classical_run)
    if args.preflight:
        if args.preflight_report is None:
            parser.error("--preflight requires an explicit fresh --preflight-report")
        if args.preflight_report.exists():
            raise FileExistsError("Do not overwrite a preflight receipt")
        receipt = preflight(specs, raw_records)
        write(args.preflight_report, receipt)
        print(json.dumps({"status": receipt["status"], "report": str(args.preflight_report),
                          "source_sha256": receipt["source_sha256"]}), flush=True)
        return
    if args.out is None or args.plan is None or not args.expected_plan_sha:
        parser.error("Run requires --out, --plan and --expected-plan-sha")
    if digest(args.plan) != args.expected_plan_sha:
        raise AssertionError("Frozen plan SHA mismatch")
    plan = read(args.plan)
    if plan.get("scientific_configuration") != scientific_configuration():
        raise AssertionError("Plan scientific_configuration must exactly match the frozen runner")
    if plan.get("source_sha256") != source_hashes():
        raise AssertionError("Plan source hashes must exactly match the two stage-f EWT sources")
    out = require_private(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("A fresh output directory is required")
    protocol = {"stage": "literature_tricks_primary_model_20261019f", "created_unix": time.time(),
                "source_sha256": source_hashes(), "dependency_sha256": dependencies,
                "plan": str(args.plan.resolve()), "plan_sha256": args.expected_plan_sha,
                "datasets": specs, "scientific_configuration": scientific_configuration(),
                "runtime": original.runtime(), "reused_raw_records": raw_records,
                "authorization_scope": "Frozen same-primary-model enhancement experiment; all new scores are exposed development"}
    write(out/"PROTOCOL_BEFORE_RUN.json", protocol)
    caches = make_caches(specs, out)
    summaries = copy_raw_references(raw_records, out)
    new_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {}
        for (market, setting), record in caches.items():
            for family in RESPONSE_FAMILIES:
                tasks[pool.submit(run_response_job, record, family, str(out))] = f"{market}/{setting}/{family}"
            tasks[pool.submit(run_ridge_job, record, str(out))] = f"{market}/{setting}/ridge_price"
        for future in as_completed(tasks):
            result = future.result()
            summaries.append(result)
            new_rows.append(result)
            write(out/"PROGRESS_SUMMARIES.json", summaries)
            print(json.dumps({"completed": tasks[future], "mean_mse_over_rw": result["mean_mse_over_rw"]}), flush=True)
    summaries.sort(key=lambda r: (r["market"], r["setting"], r["family"]))
    write(out/"ALL_SUMMARIES.json", summaries)
    unchanged = source_hashes() == protocol["source_sha256"] and verify_dependencies() == dependencies
    plan_unchanged = digest(args.plan) == args.expected_plan_sha
    completed_count = sum(r["new_fit_tasks"] for r in new_rows)
    if len(new_rows) != 30 or completed_count != 462:
        raise AssertionError("Incomplete fixed experiment")
    write(out/"EXECUTION_COMPLETE.json", {"status": "completed" if unchanged and plan_unchanged else "IDENTITY_FAILURE",
          "source_hashes_unchanged": unchanged, "plan_hash_unchanged": plan_unchanged,
          "new_prediction_sets": len(new_rows), "reused_prediction_sets": len(raw_records),
          "new_fit_tasks": completed_count, "new_response_fit_tasks": 408, "new_ridge_fit_tasks": 54,
          "selection_positive_gradient_steps": sum(r["selection_positive_gradient_steps"] for r in new_rows),
          "refit_positive_gradient_steps": sum(r["refit_positive_gradient_steps"] for r in new_rows),
          "new_initial_dynamics_diagnostic_solves": 24, "all_new_scores_development": True})
    if not unchanged or not plan_unchanged:
        raise AssertionError("Frozen source or plan changed during execution")


if __name__ == "__main__":
    main()
