"""Replay every completed v2 final checkpoint without fitting or scoring.

The frozen production constructor and saved parameter state define the model;
normalizer statistics and mature training identities are independently checked.
This script refuses an incomplete run before reading checkpoint/prediction data.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from run_factorized_experiment import construct


FREEZE_SHA = "327c96fb050e60b86cb58dfcf23c47ca971ddfe910201687cf1c54f7f8917a62"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


def compare(a, b, name):
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape {a.shape} versus {b.shape}")
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        raise AssertionError(f"{name}: NaN masks differ")
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise AssertionError(f"{name}: finite masks differ")
    maximum = float(np.max(np.abs(a[valid].astype(float)-b[valid].astype(float)))) if valid.any() else 0.0
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-9, equal_nan=True, err_msg=name)
    return {"shape": list(a.shape), "saved_dtype": str(b.dtype), "max_absolute_difference": maximum,
            "exact_equal_including_nan": bool(np.array_equal(a, b, equal_nan=True))}


def main():
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    public, private = root / "results/neural", root / "local_only/neural"
    summary = json.loads((public / "RUN_SUMMARY.json").read_text())
    if summary.get("status") != "COMPLETE" or summary.get("failures") != 0:
        raise RuntimeError("Refuse to inspect forecasts before the full run is complete")
    freeze_path = root / "PROTOCOL_FREEZE.json"
    assert sha(freeze_path) == FREEZE_SHA
    freeze = json.loads(freeze_path.read_text())
    for rel, expected in freeze["frozen_sha256"].items():
        assert sha(root / rel) == expected, rel
    cfg = json.loads((root / "experiment_config.json").read_text())
    audit = json.loads((root / "EXAMPLE_AUDIT.json").read_text())
    causal = json.loads((root / "results/FACTOR_DATA_CAUSALITY_CHECK.json").read_text())
    assert causal["status"] == "PASS" and causal["freeze_sha256"] == FREEZE_SHA
    methods, years, seeds = cfg["neural_models"], cfg["evaluation_issue_years"], cfg["refit_seeds"]
    expected_refits = len(methods)*len(years)*len(seeds)
    expected_fits = len(methods)*len(years)*(len(cfg["learning_rates"])+len(seeds))
    assert summary["models"] == methods and summary["refits"] == expected_refits == 63
    assert summary["fits"] == expected_fits == 105
    assert len(list(private.glob("*.pt"))) == expected_fits
    assert len(list(private.glob("*_refit_s*.pt"))) == expected_refits
    with np.load(root / "local_only/examples_development.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in ["x", "context", "base", "y", "anchor", "scale", "completed", "has_bid"]}
    meta = pd.read_csv(root / "local_only/example_metadata.csv")
    assert meta.target_date.max() < "2024-01-01" and meta.issue_year.max() <= 2023
    torch.set_num_threads(cfg["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    known = data["completed"].astype(bool) | data["has_bid"].astype(bool)
    records, failures = [], []
    tensor_cache = {}
    distribution_map = {
        "lognormal_mu_ratio": "lognormal_mu_ratio", "lognormal_sigma": "lognormal_sigma",
        "weights": "weights", "median_mean_z": "median_mean_z", "basis_mean_z": "basis_mean_z",
        "mechanical_failure": "mechanical_failure", "mean_bid_ratio": "mean_bid_ratio",
        "mix": "bid_mixture", "concentration": "bid_concentration",
    }
    for year in years:
        train = np.flatnonzero((meta.target_date.to_numpy() < f"{year}-01-01") & known)
        evaluation = np.flatnonzero(meta.issue_year.to_numpy() == year)
        assert not np.intersect1d(train, evaluation).size
        assert meta.target_date.iloc[train].max() < meta.issue_date.iloc[evaluation].min()
        for method in methods:
            columns = audit["price_only_columns"] if method == "GRU_price" else list(range(data["x"].shape[-1]))
            cache_key = year, tuple(columns)
            if cache_key not in tensor_cache:
                # Population moments on mature training rows only, matching the
                # documented threshold rule without invoking fit/Normalizer.
                xx = data["x"][train][:, :, columns]
                expected_normalizer = {
                    "xmean": xx.mean((0, 1), keepdims=True),
                    "xstd": xx.std((0, 1), keepdims=True),
                    "cmean": data["context"][train].mean(0, keepdims=True),
                    "cstd": data["context"][train].std(0, keepdims=True),
                }
                expected_normalizer["xstd"] = np.where(expected_normalizer["xstd"] < .05, 1., expected_normalizer["xstd"])
                expected_normalizer["cstd"] = np.where(expected_normalizer["cstd"] < .05, 1., expected_normalizer["cstd"])
                tensor_cache[cache_key] = expected_normalizer
            expected_normalizer = tensor_cache[cache_key]
            selection = json.loads((public / f"selection_{method}_{year}.json").read_text())
            for seed in seeds:
                stem = f"{method}_{year}_refit_s{seed}"
                state_path = private / f"{stem}.pt"
                prediction_path = private / f"{method}_{year}_s{seed}.npz"
                try:
                    receipt = json.loads((public / f"{stem}.json").read_text())
                    checkpoint_sha = sha(state_path)
                    assert checkpoint_sha == receipt["state_sha256"]
                    # Checkpoint is locally generated and SHA matched to its
                    # trusted run receipt before deserializing its numpy state.
                    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
                    assert checkpoint["method"] == receipt["method"] == method
                    assert checkpoint["seed"] == receipt["seed"] == seed
                    assert checkpoint["config"] == cfg
                    assert checkpoint["learning_rate"] == receipt["learning_rate"] == selection["learning_rate"]
                    assert checkpoint["selected_epoch"] == receipt["selected_epoch"] == selection["epochs"]
                    assert receipt["fit_id"] == stem and receipt["validation_MSE_RW_ratio"] is None
                    assert same(np.asarray(checkpoint["training_indices"]), train)
                    assert receipt["training_examples"] == selection["final_training_count"] == len(train)
                    norm = checkpoint["normalizer"]
                    assert list(norm["columns"]) == list(columns)
                    assert set(norm) == {"columns", "xmean", "xstd", "cmean", "cstd"}
                    for key, expected in expected_normalizer.items():
                        assert same(np.asarray(norm[key]), expected), (stem, key)
                    # Use the saved, checked training-only state for inference.
                    nx = np.clip((data["x"][:, :, columns]-norm["xmean"])/norm["xstd"], -12, 12).astype(np.float32)
                    nc = np.clip((data["context"]-norm["cmean"])/norm["cstd"], -12, 12).astype(np.float32)
                    tx, tc, tb = torch.from_numpy(nx), torch.from_numpy(nc), torch.as_tensor(data["base"], dtype=torch.float64)
                    model = construct(method, len(columns), data["context"].shape[1], cfg)
                    model.load_state_dict(checkpoint["model"], strict=True)
                    assert all(torch.equal(v.cpu(), checkpoint["model"][k].cpu()) for k, v in model.state_dict().items())
                    model.eval()
                    outputs = {}
                    with torch.no_grad():
                        for start in range(0, len(evaluation), 256):
                            batch = evaluation[start:start+256]
                            out = model(tx[batch], tc[batch], tb[batch])
                            needed = ["mean"] + (list(distribution_map) if "lognormal_mu_ratio" in out else [])
                            for key in needed:
                                outputs.setdefault(key, []).append(out[key].detach().cpu().numpy())
                    outputs = {k: np.concatenate(v) for k, v in outputs.items()}
                    with np.load(prediction_path, allow_pickle=False) as z:
                        saved = {k: z[k] for k in z.files}
                    assert same(saved["example_indices"], evaluation)
                    for saved_key, source_key in [("y_z", "y"), ("anchor", "anchor"), ("scale", "scale")]:
                        assert same(saved[saved_key], data[source_key][evaluation])
                    comparisons = {"mean_z": compare(outputs["mean"], saved["mean_z"], stem+":mean_z")}
                    if "lognormal_mu_ratio" in outputs:
                        for output_key, saved_key in distribution_map.items():
                            comparisons[saved_key] = compare(outputs[output_key], saved[saved_key], stem+":"+saved_key)
                    records.append({"fit_id": stem, "method": method, "issue_year": year, "seed": seed,
                                    "checkpoint_sha256": checkpoint_sha, "refit_receipt_sha256": sha(public / f"{stem}.json"),
                                    "prediction_sha256": sha(prediction_path), "receipt_state_sha_match": True,
                                    "checkpoint_configuration_match": True, "strict_state_load_match": True,
                                    "mature_training_indices_exact_match": True, "training_examples": len(train),
                                    "maximum_training_target_date": meta.target_date.iloc[train].max(),
                                    "minimum_evaluation_issue_date": meta.issue_date.iloc[evaluation].min(),
                                    "evaluation_examples": len(evaluation), "evaluation_excluded_from_normalizer": True,
                                    "normalizer_train_moments_exact_match": True,
                                    "fields": comparisons, "point_distribution_replay_scope": "mean only" if "lognormal_mu_ratio" not in outputs else "all saved continuous-distribution and auxiliary fields",
                                    "pass": True})
                except Exception as error:
                    failure = {"fit_id": stem, "exception_type": type(error).__name__, "reason": str(error)}
                    failures.append(failure)
                    print(json.dumps({"replay_failure": failure}), flush=True)
        print(json.dumps({"year_completed": year, "checkpoints_passed": len(records), "failures": len(failures)}), flush=True)
    after = {rel: sha(root / rel) == expected for rel, expected in freeze["frozen_sha256"].items()}
    assert all(after.values())
    all_fields = [field for record in records for field in record["fields"].values()]
    passed = not failures and len(records) == expected_refits
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if passed else "FAIL",
        "scope": "63 final model-year-seed checkpoints from the completed 105-fit run; replay only, no fitting, new tuning or outcome scoring",
        "freeze_id": freeze["freeze_id"], "freeze_sha256": FREEZE_SHA,
        "replay_script_sha256": sha(Path(__file__).resolve()), "run_summary_sha256": sha(public / "RUN_SUMMARY.json"),
        "torch_version": torch.__version__, "torch_threads": cfg["torch_threads"], "forward_batch_size": 256,
        "expected_checkpoint_count": expected_refits, "passed_checkpoint_count": len(records),
        "total_replayed_issued_forecasts": sum(r["evaluation_examples"] for r in records),
        "max_absolute_difference_all_compared_fields": max((f["max_absolute_difference"] for f in all_fields), default=None),
        "every_compared_field_exactly_equal": bool(all(f["exact_equal_including_nan"] for f in all_fields)) if passed else False,
        "normalizer_formula": "Population mean/std of mature training x across training rows and 20 positions; context over training rows; std<0.05 replaced by 1; clip normalized inputs to [-12,12]",
        "training_mask": "target_date before evaluation-year January1 and (completed or has_bid); excludes all evaluation issue rows",
        "frozen_file_checks_after_replay": after, "frozen_file_count": len(after),
        "checkpoints": records, "failures": failures, "elapsed_seconds": time.perf_counter()-started,
        "confirmation_year_contents_read": False,
        "limits": "Training replay is not an independent reimplementation of the neural forward equations; it uses the frozen constructor with saved state. Proper-score formula and auxiliary-label maturity audits are separate. Point-model residual atoms are outside this mean-only point replay scope.",
    }
    destination = root / "results/NEURAL_REPLAY_AUDIT.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "passed_checkpoint_count", "total_replayed_issued_forecasts", "max_absolute_difference_all_compared_fields", "every_compared_field_exactly_equal", "elapsed_seconds"]}), flush=True)
    if not passed:
        raise RuntimeError(f"Neural replay has {len(failures)} failures; see {destination}")


if __name__ == "__main__":
    main()
