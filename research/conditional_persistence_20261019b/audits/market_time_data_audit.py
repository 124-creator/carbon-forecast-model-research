"""Pre-fit data/time audit; no optimizer or market-score evaluation is called.

Role disclosure: this auditor implemented prepare_data.py and, at the GAS
author's request, the exact-date input guards in run_gas_benchmark.read_base.
Those checks are implementer validation. The model, normalizer, training,
selection and scoring code was written by other researchers and is reviewed
independently here. The audit is not a model-performance or publication claim.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
SOURCE = STUDY / "source"
MARKETS = ("EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI")
OLD_PANELS = tuple(f"{m}_{y}" for m in MARKETS for y in (2021, 2022, 2023)
                   if (m, y) != ("SHANGHAI", 2021))
NEW_PANELS = ("SHENZHEN_PLOS_2020", "SHENZHEN_PLOS_2021")
SOURCE_SHA = {
    "EUA_LONG": "142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f",
    "GUANGZHOU": "8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051",
    "HUBEI": "04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797",
    "SHANGHAI": "425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1",
    "SHENZHEN_PLOS": "90c49501afa9beb7370c4409084a28729daa4904a7e7e5de310d0f6bd7a3ba10",
}
H = np.array([1, 5, 20], dtype=np.int64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_close(actual, expected, *, atol=2e-11, rtol=2e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))


def load_checked(prepared_dir, panel):
    market = panel.rsplit("_", 1)[0]
    path = prepared_dir / f"{panel}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    assert receipt["cell"] == panel
    assert sha(path) == receipt["prepared_sha256"]
    assert receipt["source"]["sha256"] == SOURCE_SHA[market]
    with np.load(path, allow_pickle=False) as z:
        dates = z["all_dates"].astype("U10")
        assert len(dates) and np.all(dates[1:] > dates[:-1])
        assert np.all(dates < "2024-01-01")
        if market == "SHENZHEN_PLOS":
            assert np.all(dates < "2022-01-01")
        assert str(z["source_sha256"].item()) == SOURCE_SHA[market]
        arrays = {key: z[key].copy() for key in z.files}
    return arrays, receipt


def independently_reconstruct_preprocessing(r, dates, train, last_training_label):
    floor = max(1e-8, float(np.median(np.abs(r[1:last_training_label+1])))*0.1)
    variance = np.empty(len(r), dtype=np.float64)
    variance[0] = 0.0
    total = 0.0
    for i in range(1, len(r)):
        if i <= 20:
            total += r[i]*r[i]
            variance[i] = total/i
        else:
            variance[i] = 0.94*variance[i-1]+0.06*r[i]*r[i]
    scale = np.maximum(np.sqrt(variance), floor)
    date_numbers = dates.astype("datetime64[D]").astype(np.int64)
    gaps = np.r_[1, np.diff(date_numbers)]
    raw_features = np.column_stack((r/scale, np.log(scale), np.log1p(gaps)))
    training_windows = np.stack([raw_features[i-19:i+1] for i in train])
    mean = training_windows.reshape(-1, 3).mean(axis=0)
    std = np.maximum(training_windows.reshape(-1, 3).std(axis=0), 1e-5)
    return floor, scale, raw_features, mean, std


def audit_panel(prepared_dir, panel, gas_runner):
    d, receipt = load_checked(prepared_dir, panel)
    dates, r = d["all_dates"], d["all_returns"]
    year = panel.rsplit("_", 1)[1]
    mask = np.char.startswith(dates, year+"-")
    first = int(np.flatnonzero(mask)[0])
    expected = {
        "train": np.arange(39, first-166),
        "validation": np.arange(first-146, first-20),
        "test": np.flatnonzero(mask[:-20] & mask[20:]),
    }
    assert len(expected["train"]) >= 128
    assert len(expected["validation"]) == 126
    assert len(expected["test"]) > 0
    label_end = int(expected["train"][-1]+20)
    assert label_end == int(d["train_label_end"])
    assert label_end < expected["validation"][0]
    assert expected["validation"][-1]+20 < expected["test"][0]
    np.testing.assert_array_equal(d["horizons"], H)
    assert r[0] == 0 and len(r) == len(dates) and np.isfinite(r).all()
    floor, scale, features, mean, std = independently_reconstruct_preprocessing(r, dates, expected["train"], label_end)
    errors = {
        "training_floor": check_close(d["floor"], floor),
        "training_feature_mean": check_close(d["feature_mean"], mean),
        "training_feature_sd": check_close(d["feature_sd"], std),
    }
    role_rows = {}
    for role, ids in expected.items():
        np.testing.assert_array_equal(d[f"{role}_indices"], ids)
        np.testing.assert_array_equal(d[f"{role}_dates"], dates[ids])
        target_ids = ids[:, None]+H
        np.testing.assert_array_equal(d[f"{role}_target_indices"], target_ids)
        np.testing.assert_array_equal(d[f"{role}_target_dates"], dates[target_ids])
        assert receipt["roles"][role]["count"] == len(ids)
        future = np.stack([r[i+1:i+21] for i in ids])
        x = np.stack([features[i-19:i+1] for i in ids])
        unit = scale[ids, None]*np.sqrt(H)
        cumulative = np.stack([future[:, :h].sum(axis=1) for h in H], axis=1)
        errors[role+"_X"] = check_close(d[f"{role}_X"], (x-mean)/std)
        errors[role+"_daily_scale"] = check_close(d[f"{role}_daily_scale"], scale[ids])
        errors[role+"_future_returns"] = check_close(d[f"{role}_future_returns"], future)
        errors[role+"_future_normalized"] = check_close(d[f"{role}_future_normalized"], future/scale[ids, None])
        errors[role+"_unit"] = check_close(d[f"{role}_unit"], unit)
        errors[role+"_target"] = check_close(d[f"{role}_y"], cumulative/unit)
        assert np.all(d[f"{role}_origin_price"] > 0) and np.all(d[f"{role}_target_price"] > 0)
        observed_log_ratio = np.log(d[f"{role}_target_price"])-np.log(d[f"{role}_origin_price"][:, None])
        errors[role+"_price_log_return_identity"] = check_close(observed_log_ratio, cumulative)
        role_rows[role] = {"origins": int(len(ids)), "first_origin": int(ids[0]), "last_origin": int(ids[-1]),
                           "first_date": str(dates[ids[0]]), "last_date": str(dates[ids[-1]]),
                           "max_label_index": int(ids[-1]+20), "max_label_date": str(dates[ids[-1]+20])}
    gas_mapping = []
    if panel in OLD_PANELS and gas_runner is not None:
        # This calls only the target-process constructor, never build_prepared
        # or a normalizer optimizer. Direct convolution is a separate mapping.
        for h in (5, 20):
            observed = gas_runner.future_target_process(r, h)
            direct = np.convolve(r[1:], np.ones(h), mode="valid")
            error = check_close(observed, direct)
            fit_count = label_end-h+1
            assert (fit_count-1)+h == label_end
            for role, ids in expected.items():
                column = int(np.flatnonzero(H == h)[0])
                check_close(observed[ids], d[f"{role}_y"][:, column]*d[f"{role}_unit"][:, column])
                context_indices = ids[:, None]-h+np.arange(-19, 1)
                assert context_indices.min() >= 0
                assert np.all(context_indices[:, -1]+h == ids)
            gas_mapping.append({"h": h, "target_process_count": len(observed),
                                "training_mature_target_count": fit_count,
                                "last_fit_maturity_index": label_end,
                                "first_validation_origin": int(expected["validation"][0]),
                                "convolution_max_absolute_error": error,
                                "no_return_index0_observation": True})
    return {"panel": panel, "source_sha256": receipt["source"]["sha256"],
            "prepared_sha256": receipt["prepared_sha256"], "roles": role_rows,
            "boundary_checks_passed": 3, "max_numerical_reconstruction_error": max(errors.values()),
            "errors": errors, "gas_target_mapping": gas_mapping}


def audit_gas_causal_filter(gas):
    rng = np.random.default_rng(710261019)
    x = rng.normal(size=110)
    params = gas.GASParameters(.12, .08, .7, .6, .01, .4, .0, 1.)
    fit = gas.GASFit(params, .15, 1.1, {"origin": "pure synthetic fixed parameters; no fitting"})
    full = gas.filter_gas(x, fit)
    cut = 67
    prefix = gas.filter_gas(x[:cut], fit)
    suffix_changed = x.copy()
    suffix_changed[cut:] = rng.normal(loc=4, scale=3, size=len(x)-cut)
    changed = gas.filter_gas(suffix_changed, fit)
    errors = []
    for field in ("normalized", "prior_mean", "prior_std", "next_mean_internal", "next_scale2_internal"):
        errors.append(check_close(getattr(full, field)[:cut], getattr(prefix, field), atol=0, rtol=0))
        errors.append(check_close(getattr(full, field)[:cut], getattr(changed, field)[:cut], atol=0, rtol=0))
    for h in (1, 5, 20):
        loc, scale = gas.forecast_gas(fit, full, horizon=h)
        prefix_loc, prefix_scale = gas.forecast_gas(fit, prefix, horizon=h)
        errors.append(check_close(loc[:cut], prefix_loc, atol=0, rtol=0))
        errors.append(check_close(scale[:cut], prefix_scale, atol=0, rtol=0))
        expected_m = full.next_mean_internal.copy()
        expected_s2 = full.next_scale2_internal.copy()
        for _ in range(h-1):
            expected_m = params.omega_mean+params.beta_mean*expected_m
            expected_s2 = params.omega_scale+params.beta_scale*expected_s2
        errors.append(check_close(loc, fit.input_mean+fit.input_std*expected_m))
        errors.append(check_close(scale, fit.input_std*np.sqrt(expected_s2*params.nu/(params.nu-2))))
    return {"synthetic_only": True, "normalizer_fits": 0,
            "prefix_truncation_and_suffix_perturbation": "PASS",
            "forecast_prior_h_minus_one_pure_AR_steps": "PASS",
            "student_scale2_to_variance_conversion": "PASS", "max_error": max(errors)}


def audit_candidate_initialization(candidate):
    import torch
    torch.set_num_threads(1)
    reference = {"pi": [.2, .5, .3], "f0": .83, "rho_endpoint": .23, "rho_joint005": .79}
    x = torch.as_tensor(np.random.default_rng(910261019).normal(size=(5, 20, 3)), dtype=torch.float32)
    rows = []
    for backbone, arms in candidate.ARMS.items():
        for arm in arms:
            torch.manual_seed(31)
            model = candidate.PersistenceMarketModel(backbone, arm, reference)
            model.eval()
            with torch.no_grad():
                values = model.forecast_parameters(x)
                expected_rho = reference["rho_"+candidate.reference_mode(arm)]
                errors = [check_close(values["f"].numpy(), reference["f0"], atol=1e-13, rtol=0),
                          check_close(values["rho"].numpy(), expected_rho, atol=1e-13, rtol=0)]
                distributions = model.distributions(values)
                for h, (weights, sigmas) in distributions.items():
                    errors.append(check_close(weights.sum(-1).numpy(), 1, atol=2e-12, rtol=0))
                    errors.append(check_close((weights*sigmas**2).sum(-1).numpy(), reference["f0"]**2))
            rows.append({"backbone": backbone, "arm": arm, "matched_reference_mode": candidate.reference_mode(arm),
                         "zero_head_and_normalized_variance": "PASS", "max_error": max(errors)})
    return {"synthetic_only": True, "gradient_steps": 0, "reference_optimizer_fits": 0, "arms": rows}


def manual_code_review_snapshots(include_gas=True):
    files = ["data_sources/prepare_data.py", "source/persistence_market.py", "source/run_persistence_market.py",
             "source/persistence.py", "source/mixture_metrics.py",
             "provenance/PROTOCOL.json", "provenance/MARKET_PROTOCOL.json"]
    if include_gas:
        files += ["source/gas_filter.py", "source/run_gas_benchmark.py"]
    identities = {}
    for name in files:
        path = STUDY/name
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=name)
        identities[name] = sha(path)
    return identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("/tmp/carbon-conditional-persistence-runs/prepared"))
    parser.add_argument("--output", type=Path, default=STUDY/"audits"/"MARKET_TIME_DATA_AUDIT.json")
    parser.add_argument("--candidate-only", action="store_true", help="Freeze candidate review while GAS implementation is still separate")
    args = parser.parse_args()
    sys.path.insert(0, str(SOURCE))
    gas_runner = None if args.candidate_only else importlib.import_module("run_gas_benchmark")
    candidate = importlib.import_module("persistence_market")
    gas = None if args.candidate_only else importlib.import_module("gas_filter")
    before = manual_code_review_snapshots(include_gas=not args.candidate_only)
    panels = [audit_panel(args.prepared_dir, p, gas_runner) for p in OLD_PANELS+NEW_PANELS]
    gas_audit = {"status": "NOT_RUN_IN_CANDIDATE_ONLY_MODE"} if args.candidate_only else audit_gas_causal_filter(gas)
    candidate_audit = audit_candidate_initialization(candidate)
    after = manual_code_review_snapshots(include_gas=not args.candidate_only)
    if after != before:
        raise RuntimeError("Source changed during the audit; rerun after implementation is frozen")
    report = {
        "status": "PASS_CANDIDATE_PREFIT_DATA_TIME_CHECKS" if args.candidate_only else "PASS_PREFIT_DATA_TIME_CHECKS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only_mode": args.candidate_only,
        "audit_script_sha256": sha(__file__), "source_snapshot": after,
        "role_disclosure": {"prepared_data": "This auditor authored the preparer; numerical reconstruction is implementer validation.",
                            "GAS_read_base_guard": "Exact input split guards were added by this auditor at the author's request before fitting.",
                            "model_and_learning_time_paths": "Independent read-only review of code authored by the other two agents; no model training performed."},
        "prepared_panels": len(panels), "strict_boundaries_and_same_year_checks": 3*len(panels),
        "panels": panels, "gas_causal_filter": gas_audit, "candidate_initialization": candidate_audit,
        "manual_time_review": {
            "candidate_reference_fit": "Only train_future_normalized and train_y reach fit_references; pi/f0 uses h1 training targets; rho fits use training endpoint/path losses.",
            "candidate_gradient_fit": "fit_neural constructs tensors for train and validation only; optimizer minibatches use train; epoch0 plus all40 epochs are considered by validation equal-horizon CRPS.",
            "candidate_deployment": "Reload selected weights once; predict is eval/no_grad and takes only role_X; no test parameter update. Train/validation/test arrays share fixed train-only preprocessing.",
            "gas_static_fit": "Target x[j] matures at j+h. x[:train_label_end-h+1] is the only fit prefix; all static standardization/initial parameters use that prefix.",
            "gas_filter_deployment": "The full prefix scan is causal given frozen parameters; at origin o only observations through x[o-h] are used; the target is x[o], mature at o+h.",
            "gas_neural_fit": "Train minibatches only; validation includes epoch0 and selects earliest strict CRPS improvement after all40 epochs; test prediction follows selected-state load.",
            "gas_controls": "Empirical atoms use training forward residuals only; no-neural distributions have frozen static normalization; common daily-EWMA scoring units are identical for all compared arms.",
            "source_strata": "GAS only11 existing panels; candidate13, with Shenzhen2020/2021 reported separately. No2024+ arrays were loaded.",
        },
        "market_model_fits": 0, "market_gradient_steps": 0, "market_test_scores_computed": 0,
        "reserved2024plus_price_access": False,
        "limits": ["This pre-fit audit does not prove future execution completeness, checkpoint identity or model performance.",
                   "Execution receipts and saved predictions require a separate post-run replay/consistency audit.",
                   "The two newly used Shenzhen years are not two independent markets; all old development panels have prior exposure."],
    }
    if args.candidate_only:
        report["manual_time_review"] = {key: value for key, value in report["manual_time_review"].items()
                                        if not key.startswith("gas_")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "panels": len(panels), "maturity_checks": 3*len(panels),
                      "normalizer_or_market_fits": 0, "market_test_scores_computed": 0,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
