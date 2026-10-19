#!/usr/bin/env python3
"""Replay all fixed development checkpoints and independently audit scores.

No experiment-source module is imported, no fitting routine is called, and no
source CSV or reserved-period file is opened. Inputs are the already prepared
development arrays, archived checkpoints/predictions, and public receipts.
Outputs contain hashes, discrepancies and aggregate metrics only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
from scipy.special import erf, ndtr
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PRIMARY = "path_block_gate3_delta005"
RAW = "raw_block_gate3_same_joint_budget"
NO_PREFIX = "path_no_prefix_gate3_delta005"
BACKBONES = ("MLP32_16", "GRU16")
SEEDS = (17, 41, 89)
REPEATS, BOOTSTRAP_SEED, TIE_TOLERANCE = 1999, 219031, 1e-10
METRIC_NAMES = ("normalized_CRPS", "price_CRPS", "mean_price_squared_error", "coverage90", "width90", "alpha")
ORIGINAL_DATE_GUARD = '''    df = pd.read_csv(data_path)
    dates = pd.to_datetime(df["date"])
    price = df["price"].to_numpy(float)
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("This development runner refuses reserved-period prices")
'''
CORRECTED_DATE_GUARD = '''    date_only = pd.read_csv(data_path, usecols=["date"])
    dates = pd.to_datetime(date_only["date"])
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("This development runner refuses reserved-period prices")
    df = pd.read_csv(data_path)
    price = df["price"].to_numpy(float)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class IndependentScaleReplay(nn.Module):
    """Architecture-only reconstruction, independently defined in this audit."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        if backbone == "MLP32_16":
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(60, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh())
        elif backbone == "GRU16":
            self.encoder = nn.GRU(3, 16, batch_first=True)
        else:
            raise ValueError(backbone)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        if self.backbone == "MLP32_16":
            encoded = self.encoder(x)
        else:
            _, hidden = self.encoder(x)
            encoded = hidden[-1]
        return .5 + torch.sigmoid(self.head(encoded).squeeze(-1))


def normal_abs(value, scale):
    value, scale = np.broadcast_arrays(value, scale)
    z = value / scale
    return math.sqrt(2 / math.pi) * scale * np.exp(-z*z/2) + value * erf(z / math.sqrt(2))


def normal_score(y, weights, scales):
    w, s = np.broadcast_arrays(weights, scales)
    absolute = np.sum(w * normal_abs(np.asarray(y)[..., None], s), axis=-1)
    pair_scales = np.hypot(s[..., :, None], s[..., None, :])
    pair_mean = math.sqrt(2 / math.pi) * np.sum(w[..., :, None] * w[..., None, :] * pair_scales, axis=(-2, -1))
    return absolute - pair_mean / 2


def ab_terms(y, weights, s0, factor):
    n = len(y)
    s0 = np.broadcast_to(s0, (n, len(weights)))
    s1 = s0 * factor[:, None]
    w = np.broadcast_to(weights, s0.shape)
    def distance(left, right):
        return math.sqrt(2 / math.pi) * np.sum(w[:, :, None] * w[:, None, :] * np.hypot(left[:, :, None], right[:, None, :]), axis=(1, 2))
    a = distance(s0, s1) - (distance(s0, s0) + distance(s1, s1)) / 2
    score0, score1 = normal_score(y, w, s0), normal_score(y, w, s1)
    b = (a + score0 - score1) / 2
    return a, b, score0, score1


def price_score_and_mean(y, mu, sig, weights):
    """Independent expected-absolute-distance formula, prechecked by quadrature."""
    mu, sig, w = np.broadcast_arrays(mu, sig, weights)
    mean_parts = np.exp(mu + sig*sig/2)
    assert np.all(np.isfinite(mean_parts)) and np.all(mean_parts > 0)
    assert np.all(np.asarray(y) > 0) and np.all(sig > 0)
    logy = np.log(np.asarray(y))[..., None]
    first = (mean_parts - np.asarray(y)[..., None]) + 2 * (
        np.asarray(y)[..., None] * ndtr((logy-mu)/sig) - mean_parts * ndtr((logy-mu-sig*sig)/sig))
    den = np.hypot(sig[..., :, None], sig[..., None, :])
    # E min(X_i,X_j) uses two truncated first moments under independent logs.
    min_mean = mean_parts[..., :, None] * ndtr((mu[..., None, :]-mu[..., :, None]-sig[..., :, None]**2)/den)
    min_mean += mean_parts[..., None, :] * ndtr((mu[..., :, None]-mu[..., None, :]-sig[..., None, :]**2)/den)
    pair = mean_parts[..., :, None] + mean_parts[..., None, :] - 2*min_mean
    score = np.sum(w*first, axis=-1) - .5*np.sum(w[..., :, None]*w[..., None, :]*pair, axis=(-2, -1))
    return score, np.sum(w*mean_parts, axis=-1)


def atomic_score(y, atoms):
    """Exact midpoint pinball identity, distinct from source's sorted-sum code."""
    x = np.sort(atoms, axis=-1)
    tau = (np.arange(x.shape[-1])+.5)/x.shape[-1]
    residual = np.asarray(y)[..., None]-x
    return 2*np.mean(np.maximum(tau*residual, (tau-1)*residual), axis=-1)


def differences(name, actual, expected, maxima, *, atol=1e-9, rtol=2e-10):
    a, b = np.broadcast_arrays(np.asarray(actual), np.asarray(expected))
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), name + " nonfinite"
    error = float(np.max(np.abs(a-b), initial=0))
    maxima[name] = max(maxima.get(name, 0), error)
    assert np.allclose(a, b, atol=atol, rtol=rtol), f"{name}: max abs discrepancy={error:.12g}"


def check_prepared(cell, data, receipt, maxima):
    year, horizon = int(cell.rsplit("_", 2)[-2]), int(cell.rsplit("_h", 1)[1])
    assert year <= 2023 and horizon in (5, 20)
    roles = ("train", "validation", "calibration", "test")
    for role in roles:
        ix = data[f"{role}_indices"]
        assert len(ix) > 0 and np.all(np.diff(ix) > 0)
        assert data[f"{role}_X"].shape == (len(ix), 20, 3)
        assert np.all(data[f"{role}_dates"] < "2024-01-01")
        assert receipt["roles"][role]["last_label_date"] < "2024-01-01"
        assert receipt["roles"][role]["count"] == len(ix)
        assert receipt["roles"][role]["first_origin"] == int(ix[0])
        assert receipt["roles"][role]["last_origin"] == int(ix[-1])
        assert receipt["roles"][role]["last_label_index"] == int(ix[-1]+horizon)
        differences("path_sum_to_endpoint", data[f"{role}_path"].sum(axis=1), data[f"{role}_y"], maxima)
        target = np.log(data[f"{role}_target_price"]/data[f"{role}_origin_price"])/data[f"{role}_unit"]
        differences("log_price_target_to_normalized_label", target, data[f"{role}_y"], maxima)
        assert np.all(data[f"{role}_basis"] >= 0)
        differences("basis_partition", data[f"{role}_basis"].sum(axis=1), 1., maxima, atol=1e-12)
    for left, right in zip(roles[:-1], roles[1:]):
        assert data[f"{left}_indices"][-1]+horizon < data[f"{right}_indices"][0]
    assert len(data["train_indices"]) >= 128
    assert len(data["validation_indices"]) == 126 and len(data["calibration_indices"]) == 252
    assert np.all(np.char.startswith(data["test_dates"], str(year)))
    assert np.all(data["feature_sd"] > 0)
    differences("standardized_train_feature_mean", data["train_X"].mean(axis=(0, 1)), 0., maxima, atol=1e-10)
    differences("standardized_train_feature_sd", data["train_X"].std(axis=(0, 1)), 1., maxima, atol=1e-8)


def verify_job_scores(data, prediction, receipt, maxima):
    names = prediction["methods"].tolist()
    index = {name: j for j, name in enumerate(names)}
    values = prediction["metrics"]
    assert values.shape == (len(data["test_y"]), 21, 6)
    assert tuple(prediction["metric_names"]) == METRIC_NAMES
    assert np.all(np.isfinite(values[:, :, :5]))
    w, s0 = data["weights"], data["sigma0"]
    independent_ab = {}
    for role in ("calibration", "test"):
        factor = prediction[f"factors_{role}"]
        a, b, score0, score1 = ab_terms(data[f"{role}_y"], w, s0, factor)
        differences("independent_a_"+role, a, prediction[f"a_{role}"], maxima, atol=2e-12)
        differences("independent_b_"+role, b, prediction[f"b_{role}"], maxima, atol=2e-12)
        independent_ab[role] = a, b, score0, score1
    y, unit, price0, target = (data["test_"+field] for field in ("y", "unit", "origin_price", "target_price"))
    factor = prediction["factors_test"]
    a, b, score0, _ = independent_ab["test"]
    gaussian_rows, atomic_rows = 0, 0
    for j, name in enumerate(names):
        if name in ("quantile_BOAG_online", "FHS"):
            atoms = prediction["issued_quantile_atoms"] if name == "quantile_BOAG_online" else np.broadcast_to(data["fhs_atoms"], (len(y), 101))
            assert np.all(np.diff(atoms, axis=1) >= 0)
            price_atoms = price0[:, None]*np.exp(unit[:, None]*atoms)
            assert np.all(np.isfinite(price_atoms)) and np.all(price_atoms > 0)
            normscore = atomic_score(y, atoms)
            pscore = atomic_score(target, price_atoms)
            mean = price_atoms.mean(axis=1)
            low = price_atoms[:, int(math.ceil(.05*atoms.shape[1]))-1]
            high = price_atoms[:, int(math.ceil(.95*atoms.shape[1]))-1]
            differences("atomic_coverage", ((target >= low)&(target <= high)).astype(float), values[:, j, 3], maxima, atol=0, rtol=0)
            differences("atomic_width", high-low, values[:, j, 4], maxima)
            assert np.all(np.isnan(values[:, j, 5]))
            atomic_rows += len(y)
        else:
            if name in ("EWMA_Normal", "GARCH_Normal"):
                mw = np.ones((len(y), 1))
                ms = np.ones((len(y), 1)) if name == "EWMA_Normal" else data["test_garch_sigma"][:, None]
                assert np.all(np.isnan(values[:, j, 5]))
            else:
                alpha = values[:, j, 5]
                assert np.all(np.isfinite(alpha)) and np.all((alpha >= 0)&(alpha <= 1))
                mw = np.concatenate(((1-alpha[:, None])*w, alpha[:, None]*w), axis=1)
                ms = np.concatenate((np.broadcast_to(s0, (len(y), len(w))), factor[:, None]*s0), axis=1)
                differences("a_b_quadratic_all_mixture_scores", score0+a*alpha**2-2*b*alpha, values[:, j, 0], maxima)
                if name in receipt["calibration_gates"]:
                    coefficients = np.asarray(receipt["calibration_gates"][name]["weights"])
                    basis = np.ones((len(y), 1)) if coefficients.size == 1 else data["test_basis"]
                    differences("saved_gate_weights_to_issued_alpha", basis@coefficients, alpha, maxima, atol=2e-12)
                if name == "F0":
                    assert np.all(alpha == 0)
                if name == "F1":
                    assert np.all(alpha == 1)
                if name == "equal_mixture":
                    assert np.all(alpha == .5)
            normscore = normal_score(y, mw, ms)
            pscore, mean = price_score_and_mean(target, np.log(price0)[:, None], unit[:, None]*ms, mw)
            gaussian_rows += len(y)
        differences("all_normalized_CRPS_external_formula", normscore, values[:, j, 0], maxima)
        differences("all_price_CRPS_external_formula", pscore, values[:, j, 1], maxima, atol=3e-8)
        differences("all_price_mean_squared_error", (mean-target)**2, values[:, j, 2], maxima, atol=3e-8)
    differences("random_walk_squared_error", (target-price0)**2, prediction["rw_squared_error"], maxima)
    assert np.array_equal(prediction["test_indices"], data["test_indices"])
    assert np.all(prediction["spent"] <= prediction["cap"]+1e-12)
    assert np.all(np.abs(prediction["control"]) <= prediction["cap"]+1e-12)
    expected_cap = .25*math.sqrt(2/math.pi)*np.dot(w, s0)*np.abs(prediction["factors_calibration"]-1)
    differences("recorded_control_cap", expected_cap, prediction["cap"], maxima, atol=1e-12)
    return gaussian_rows, atomic_rows


def circular_frequency(n, length, repeats, rng):
    """Same declared circular blocks; independently reduce to origin frequencies."""
    starts = rng.integers(n, size=(repeats, math.ceil(n/length)))
    counts = np.zeros((repeats, n))
    rows = np.arange(repeats)
    filled = 0
    for block in range(starts.shape[1]):
        take = min(length, n-filled)
        for offset in range(take):
            counts[rows, (starts[:, block]+offset)%n] += 1
        filled += take
    assert np.all(counts.sum(axis=1) == n)
    return counts/n


def compare_record(point_by_cell, bootstrap, expected, comparator, maxima, key):
    mean = float(np.mean(point_by_cell))
    interval = np.quantile(bootstrap, [.025, .975])
    better, worse = int(np.sum(point_by_cell < -TIE_TOLERANCE)), int(np.sum(point_by_cell > TIE_TOLERANCE))
    tied = len(point_by_cell)-better-worse
    differences(key+"_mean", mean, expected["mean_cell_relative_difference"], maxima, atol=1e-12)
    differences(key+"_interval", interval, expected["paired_within_cell_block_interval95"], maxima, atol=1e-12)
    assert (better, tied, worse) == (expected["cells_better"], expected["cells_tied"], expected["cells_worse"])
    assert expected["n_cells"] == len(point_by_cell) and expected["comparator"] == comparator
    return {"mean_cell_relative_difference": mean, "paired_within_cell_block_interval95": interval.tolist(),
            "cells_better": better, "cells_tied": tied, "cells_worse": worse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, default=Path("/tmp/carbon-distribution-update-runs"))
    parser.add_argument("--public-dir", type=Path, default=EXPERIMENT/"results"/"market")
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(1)
    local, public = args.local_dir, args.public_dir
    summary = read_json(public/"SUMMARY.json")
    prefit = read_json(public/"MARKET_PREFIT.json")
    full_receipts = {r["job"]: r for r in read_json(public/"FIT_AND_GATE_RECEIPTS.json")}
    preparations = {r["cell"]: r for r in read_json(public/"PREPARATION_RECEIPTS.json")}
    completion = {r["job"]: r for r in read_json(public/"JOB_COMPLETION.json")}
    public_cell = {(r["cell"], r["method"]): r for r in read_json(public/"CELL_METRICS.json")}
    expected_cells = sorted(f"{m}_{y}_h{h}" for m in ("EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI") for y in (2021, 2022, 2023) for h in (5, 20) if (m, y) != ("SHANGHAI", 2021))
    assert sorted(preparations) == expected_cells
    assert sorted(f"{m}_{y}_h{h}" for m, y, h in prefit["cells"]) == expected_cells
    expected_jobs = sorted(f"{c}_{b}_s{s}" for c in expected_cells for b in BACKBONES for s in SEEDS)
    assert sorted(full_receipts) == expected_jobs and sorted(completion) == expected_jobs
    assert sorted(p.stem for p in (local/"checkpoints").glob("*.pt")) == expected_jobs
    assert sorted(p.stem for p in (local/"predictions").glob("*.npz")) == expected_jobs
    assert sorted(p.stem for p in (local/"prepared").glob("*.npz")) == expected_cells
    source_hashes = {name: sha(EXPERIMENT/"source"/name) for name in prefit["source_sha256"]}
    source_identity_changes = {name: {"frozen": prefit["source_sha256"][name], "current": value}
                               for name, value in source_hashes.items() if value != prefit["source_sha256"][name]}
    assert set(source_identity_changes).issubset({"run_market.py"}), "A fitted model/score module changed after freeze"
    source_change_proof = {}
    if source_identity_changes:
        runner = (EXPERIMENT/"source"/"run_market.py").read_text(encoding="utf-8")
        assert runner.count(CORRECTED_DATE_GUARD) == 1
        restored = runner.replace(CORRECTED_DATE_GUARD, ORIGINAL_DATE_GUARD, 1)
        restored_hash = hashlib.sha256(restored.encode("utf-8")).hexdigest()
        assert restored_hash == prefit["source_sha256"]["run_market.py"], "Runner change exceeds declared date-only guard correction"
        fix_path = EXPERIMENT/"provenance"/"POST_START_AUDIT_FIXES.json"
        fixes = read_json(fix_path)
        declared = [c for c in fixes["changes"] if c["file"] == "source/run_market.py"]
        assert len(declared) == 1 and declared[0]["refits_caused"] == 0
        source_change_proof = {"kind": "Date-only guard before price conversion, declared after the 22 prepared panels were complete.",
            "exact_reverse_patch_sha256": restored_hash,
            "reverse_patch_recovers_every_frozen_byte": True,
            "post_start_fix_receipt_sha256": sha(fix_path),
            "computational_effect_on_prepared_2023_or_earlier_arrays": "None; existing arrays and all checkpoint/score hashes independently verified below.",
            "refits": 0}
    assert sha(EXPERIMENT/"provenance"/"PROTOCOL.json") == prefit["protocol_sha256"]
    assert sha(EXPERIMENT/"provenance"/"DESIGN_ADDENDUM_PRE_RUN.json") == prefit["addendum_sha256"]
    maxima, manifests, selected_epochs = {}, [], []
    cell_arrays, backbone_arrays, cell_means, rw_means = {}, {}, {}, {}
    steps, training_epochs, replayed_factors, gaussian_rows, atomic_rows = 0, 0, 0, 0, 0
    changed_jobs, active_gate_jobs, control_selected_jobs = [], [], []
    changed_history_count = 0
    methods = None
    for cell_number, cell in enumerate(expected_cells, 1):
        prepared_path = local/"prepared"/(cell+".npz")
        prepared_hash = sha(prepared_path)
        assert prepared_hash == preparations[cell]["prepared_sha256"]
        assert read_json(local/"prepared"/(cell+".json")) == preparations[cell]
        with np.load(prepared_path, allow_pickle=False) as loaded:
            data = dict(loaded)
        check_prepared(cell, data, preparations[cell], maxima)
        cell_jobs, by_backbone = [], defaultdict(list)
        horizon = int(cell.rsplit("_h", 1)[1])
        for backbone in BACKBONES:
            for seed in SEEDS:
                job = f"{cell}_{backbone}_s{seed}"
                checkpoint_path, prediction_path = local/"checkpoints"/(job+".pt"), local/"predictions"/(job+".npz")
                receipt = read_json(local/"receipts"/(job+".json"))
                assert {k: v for k, v in receipt.items() if k != "metrics"} == full_receipts[job]
                checkpoint_hash, prediction_hash = sha(checkpoint_path), sha(prediction_path)
                assert checkpoint_hash == receipt["checkpoint_sha256"]
                assert prediction_hash == receipt["predictions_sha256"]
                ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                assert ck["backbone"] == backbone and ck["seed"] == seed
                for field in ("weights", "sigma0", "feature_mean", "feature_sd"):
                    assert np.array_equal(ck[field], data[field]), f"{job}: checkpoint {field} mismatch"
                assert all(bool(torch.isfinite(v).all()) for v in ck["state_dict"].values())
                model = IndependentScaleReplay(backbone)
                model.load_state_dict(ck["state_dict"], strict=True)
                model.eval()
                with np.load(prediction_path, allow_pickle=False) as loaded:
                    prediction = dict(loaded)
                if methods is None:
                    methods = prediction["methods"].tolist()
                    assert len(methods) == len(set(methods)) == 21
                assert methods == prediction["methods"].tolist()
                for role in ("calibration", "test"):
                    with torch.inference_mode():
                        replay = model(torch.as_tensor(data[role+"_X"], dtype=torch.float32)).numpy().astype(float)
                    differences("checkpoint_"+role+"_factor", replay, prediction["factors_"+role], maxima, atol=1e-7, rtol=0)
                    assert np.all((replay >= .5)&(replay <= 1.5))
                    replayed_factors += len(replay)
                fit = receipt["fit"]
                assert fit["actual_training_epochs"] == 60
                assert fit["gradient_optimizer_steps"] == 60*math.ceil(len(data["train_X"])/64)
                assert [r["epoch"] for r in fit["history"]] == list(range(61))
                assert all(np.isfinite(r["validation_CRPS"]) for r in fit["history"])
                best_epoch, best_score = 0, fit["history"][0]["validation_CRPS"]
                for item in fit["history"][1:]:
                    if item["validation_CRPS"] < best_score-1e-8:
                        best_epoch, best_score = item["epoch"], item["validation_CRPS"]
                assert best_epoch == fit["selected_epoch"] == ck["selected_epoch"] == completion[job]["selected_epoch"]
                assert best_score == fit["selected_validation_CRPS"]
                assert completion[job]["status"] == "completed"
                assert fit["parameters"] == sum(v.numel() for v in ck["state_dict"].values())
                changed_history_count += len(set(r["validation_CRPS"] for r in fit["history"])) > 1
                if best_epoch == 0:
                    assert bool(torch.count_nonzero(ck["state_dict"]["head.weight"]) == 0)
                    assert bool(torch.count_nonzero(ck["state_dict"]["head.bias"]) == 0)
                    assert np.all(prediction["factors_calibration"] == 1) and np.all(prediction["factors_test"] == 1)
                selected_epochs.append(best_epoch)
                steps += fit["gradient_optimizer_steps"]
                training_epochs += fit["actual_training_epochs"]
                count_gauss, count_atoms = verify_job_scores(data, prediction, receipt, maxima)
                gaussian_rows += count_gauss; atomic_rows += count_atoms
                origins = prediction["test_indices"]
                mature_count = int(np.sum(origins+horizon <= origins[-1]))
                assert receipt["online_updates_with_verified_maturity"] == mature_count
                assert receipt["online_last_observed_origin"] == int(origins[mature_count-1]) if mature_count else receipt["online_last_observed_origin"] is None
                index = {name: j for j, name in enumerate(methods)}
                if np.max(np.abs(prediction["metrics"][:, index[PRIMARY], 5]-prediction["metrics"][:, index[RAW], 5]), initial=0) > 1e-12:
                    changed_jobs.append(job)
                gate = receipt["calibration_gates"][PRIMARY]
                if np.any(np.asarray(gate["weights"]) > 1e-12): active_gate_jobs.append(job)
                if np.any(np.asarray(gate["selected_lambdas"]) > 0): control_selected_jobs.append(job)
                for j, name in enumerate(methods):
                    for k, metric in enumerate(METRIC_NAMES[:5]):
                        differences("private_receipt_metric_means", prediction["metrics"][:, j, k].mean(), receipt["metrics"][name][metric], maxima)
                cell_jobs.append(prediction["metrics"])
                by_backbone[backbone].append(prediction["metrics"])
                manifests.append({"job": job, "checkpoint_sha256": checkpoint_hash, "prediction_sha256": prediction_hash,
                                  "prepared_sha256": prepared_hash, "selected_epoch": best_epoch,
                                  "actual_training_epochs": 60, "optimizer_steps": fit["gradient_optimizer_steps"],
                                  "calibration_factor_count": len(prediction["factors_calibration"]), "test_factor_count": len(origins)})
        assert len(cell_jobs) == 6
        averaged = np.mean(np.stack(cell_jobs), axis=0)
        cell_arrays[cell] = averaged
        cell_means[cell] = averaged[:, :, :5].mean(axis=0)
        backbone_arrays[cell] = {b: np.mean(np.stack(v), axis=0) for b, v in by_backbone.items()}
        rw_means[cell] = float(np.mean((data["test_target_price"]-data["test_origin_price"])**2))
        index = {name: j for j, name in enumerate(methods)}
        for j, name in enumerate(methods):
            row = public_cell[cell, name]
            actual = {"CRPS_normalized": cell_means[cell][j, 0], "CRPS_ratio_F0": cell_means[cell][j, 0]/cell_means[cell][index["F0"], 0],
                "price_CRPS": cell_means[cell][j, 1], "price_CRPS_ratio_F0": cell_means[cell][j, 1]/cell_means[cell][index["F0"], 1],
                "price_mean_MSE_ratio_RW": cell_means[cell][j, 2]/rw_means[cell], "coverage90": cell_means[cell][j, 3],
                "interval_width_price_units": cell_means[cell][j, 4]}
            for metric, value in actual.items(): differences("public_cell_metrics", value, row[metric], maxima)
            assert row["n_origins"] == len(averaged)
        print(json.dumps({"replayed_cells": cell_number, "total_cells": len(expected_cells), "replayed_checkpoints": len(manifests)}), flush=True)
    assert len(public_cell) == len(expected_cells)*len(methods)
    index = {name: j for j, name in enumerate(methods)}
    per_cell = np.stack([cell_means[c] for c in expected_cells])
    point_raw = per_cell[:, :, 0]/per_cell[:, index[RAW]:index[RAW]+1, 0]-1
    point_f0 = per_cell[:, :, 0]/per_cell[:, index["F0"]:index["F0"]+1, 0]-1
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_raw, boot_f0, boot_no_prefix = np.zeros((REPEATS, len(methods))), np.zeros((REPEATS, len(methods))), np.zeros(REPEATS)
    for cell in expected_cells:
        arr = cell_arrays[cell][:, :, 0]
        freq = circular_frequency(len(arr), 2*int(cell.rsplit("_h", 1)[1]), REPEATS, rng)
        means = freq @ arr
        boot_raw += (means/means[:, index[RAW]:index[RAW]+1]-1)/len(expected_cells)
        boot_f0 += (means/means[:, index["F0"]:index["F0"]+1]-1)/len(expected_cells)
        boot_no_prefix += (means[:, index[PRIMARY]]/means[:, index[NO_PREFIX]]-1)/len(expected_cells)
    expected_method = {r["method"]: r for r in summary["method_results"]}
    method_results = []
    for j, name in enumerate(methods):
        relative_raw = compare_record(point_raw[:, j], boot_raw[:, j], summary["all_vs_same_joint_raw"][name], RAW, maxima, "summary_raw")
        relative_f0 = compare_record(point_f0[:, j], boot_f0[:, j], expected_method[name], "F0", maxima, "summary_F0")
        ratios = {"mean_cell_CRPS_ratio_F0": float(np.mean(per_cell[:, j, 0]/per_cell[:, index["F0"], 0])),
                  "mean_cell_price_CRPS_ratio_F0": float(np.mean(per_cell[:, j, 1]/per_cell[:, index["F0"], 1])),
                  "mean_cell_price_mean_MSE_ratio_RW": float(np.mean(per_cell[:, j, 2]/np.array([rw_means[c] for c in expected_cells]))),
                  "mean_cell_coverage90": float(np.mean(per_cell[:, j, 3]))}
        for key, value in ratios.items(): differences("summary_secondary_ratios", value, expected_method[name][key], maxima, atol=1e-12)
        method_results.append({"method": name, "vs_same_joint_raw": relative_raw, "vs_F0": relative_f0, **ratios})
    primary = compare_record(point_raw[:, index[PRIMARY]], boot_raw[:, index[PRIMARY]], summary["primary_vs_raw"], RAW, maxima, "primary")
    point_np = per_cell[:, index[PRIMARY], 0]/per_cell[:, index[NO_PREFIX], 0]-1
    compare_record(point_np, boot_no_prefix, summary["primary_vs_no_prefix"], NO_PREFIX, maxima, "primary_no_prefix")
    improving = [{"cell": cell, "relative_difference": float(point_raw[i, index[PRIMARY]]),
                  "primary_CRPS": float(per_cell[i, index[PRIMARY], 0]), "same_joint_raw_CRPS": float(per_cell[i, index[RAW], 0])}
                 for i, cell in enumerate(expected_cells) if point_raw[i, index[PRIMARY]] < -TIE_TOLERANCE]
    assert len(improving) == 1 and improving[0]["cell"] == "GUANGZHOU_2021_h20"
    assert sorted(changed_jobs) == sorted(summary["control_changed_jobs"])
    independent_counts = {"markets": 4, "market_year_horizon_cells": len(expected_cells), "neural_gradient_fits": len(manifests),
        "global_mixture_fits": len(preparations), "GARCH_fits": len(preparations), "methods": len(methods),
        "forecast_score_streams": len(manifests)*len(methods), "distinct_within_cell_evaluation_origins": sum(len(v) for v in cell_arrays.values()),
        "selected_epoch_zero": selected_epochs.count(0), "optimizer_steps": steps, "nominal_training_epochs": training_epochs,
        "main_nonzero_gate_fits": len(active_gate_jobs), "main_nonzero_control_selected_fits": len(control_selected_jobs),
        "main_control_changes_gate_vs_same_joint_raw": len(changed_jobs)}
    assert independent_counts == summary["counts"], (independent_counts, summary["counts"])
    assert steps == 116640 and selected_epochs.count(0) == 46 and training_epochs == 7920
    assert changed_history_count == 132
    gate_result = {"paired_interval_below_zero": primary["paired_within_cell_block_interval95"][1] < 0,
                  "at_least_60pct_cells_improve": primary["cells_better"]/22 >= .6,
                  "not_worse_than_no_prefix": float(np.mean(point_np)) <= 0}
    assert gate_result == summary["advance_gates"]
    assert not all(gate_result.values()) and summary["decision"] == "STOP_FIXED_CANDIDATE_NO_CONFIRMATION_OPENED"
    assert summary["new_confirmation_values_opened"] is False
    all_passed = not source_identity_changes or bool(source_change_proof.get("reverse_patch_recovers_every_frozen_byte"))
    output = {"status": "passed" if all_passed else "numerical_pass_pending_runner_identity", "date": "2026-10-19", "all_passed": all_passed,
        "scope": {"new_predictive_fits": 0, "new_gradient_optimizer_steps": 0, "reserved_files_read": 0, "source_csv_files_read": 0,
                  "experiment_modules_imported": [], "analyze_results_imported": False,
                  "checkpoints_replayed": len(manifests), "factor_values_replayed": replayed_factors,
                  "gaussian_score_rows_independently_checked": gaussian_rows, "atomic_score_rows_independently_checked": atomic_rows},
        "counts_reconciled": independent_counts, "fit_histories_nonconstant": changed_history_count,
        "maximum_absolute_discrepancies": maxima, "primary_vs_same_joint_raw": primary,
        "improving_cells": improving, "gate_decision": gate_result, "decision": summary["decision"],
        "method_results": method_results,
        "bootstrap": {"replicates": REPEATS, "seed": BOOTSTRAP_SEED, "cell_order": expected_cells,
                      "block_length": "2*h", "construction": "Circular blocks reduced independently to origin frequencies; method/learner pairing preserved.",
                      "quantile": "linear [.025,.975]", "tie_tolerance": TIE_TOLERANCE},
        "interpretation": {"aggregation": "Within each fixed cell, average each origin's six learner scores; this is not the CRPS of an ensemble CDF. Then average cell-level ratios equally.",
            "interval": "Resampling occurs only within the 22 fixed cells. It does not sample markets or capture cross-cell dependence, and does not establish transferable gains across markets or future shifts.",
            "epoch_zero": "46 selected initialization checkpoints are validation fallback after 60 logged training epochs per job; they are not zero-training jobs. All 132 histories contain 61 epoch records and satisfy 60*ceil(n_train/64) optimizer-step accounting.",
            "control_selection_counts": "63 fits select a nonzero lambda in at least one gate coordinate, which does not imply an active update. Only 14 main gates are nonzero, and only 6 issued gate paths differ from same-joint raw, all in one cell.",
            "fit_evidence": "Training work is reconciled from the archived full histories, job receipts, source and checkpoint hashes; the audit does not claim to have re-executed or independently observed every historical gradient step.",
            "limits": "Only the supplied prepared development dataset is opened. Matching scaler state and standardized training moments do not independently re-authenticate source prices. Normal-mixture coverage and width are aggregated as archived; normalized CRPS, price CRPS and price mean MSE are recomputed for every method and origin."},
        "identity": {"script_sha256": sha(__file__), "summary_sha256": sha(public/"SUMMARY.json"),
                     "cell_metrics_sha256": sha(public/"CELL_METRICS.json"), "fit_receipts_sha256": sha(public/"FIT_AND_GATE_RECEIPTS.json"),
                     "current_source_sha256": source_hashes, "frozen_source_sha256": prefit["source_sha256"],
                     "observed_post_start_source_identity_changes": source_identity_changes,
                     "post_start_source_change_proof": source_change_proof,
                     "unresolved_source_identity_changes": {}, "checkpoint_prediction_manifest": manifests},
        "audit_development_record": {"first_invocation": "Stopped at the runner hash mismatch before checkpoint replay.",
            "second_invocation": "Completed all numerical checks, retained numerical_pass_pending_runner_identity rather than silently passing the changed file.",
            "final_resolution": "Read the declared post-start date guard fix and independently reverse only that exact block; SHA256 matches the original frozen bytes. Final replay repeats verification only, never a fit."},
        "elapsed_seconds": time.time()-started}
    outpath = HERE/"INDEPENDENT_MARKET_REPLAY.json"
    outpath.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    report = ["# 市场实验独立重放与结果核验", "", f"日期：2026-10-19。状态：{'总体通过' if all_passed else '数值通过，runner 身份变化尚待解释'}。未重新拟合，未读取源 CSV 或保留期文件。", "",
        "## 结论", "", f"132 个 checkpoint 的全部 calibration/test factor 共 {replayed_factors:,} 个数值完成重放；独立定义 MLP/GRU 架构并加载保存权重，没有 import 实验源码或 analyze_results.py。", "",
        f"全部 21 个方法、22 个 cell、6 个 learner 的 normalized CRPS、price CRPS 与价格均值 MSE逐点外部公式核验通过，累计 {gaussian_rows+atomic_rows:,} 条方法评分。原子分布使用等权中点 pinball 恒等式；Gaussian/lognormal 使用已通过外部积分审计的数学公式。", "",
        f"主候选相对同等 joint budget 的 raw 对照，22 个 cell 等权相对差为 **{primary['mean_cell_relative_difference']:.14f}**，即 **{100*primary['mean_cell_relative_difference']:.8f}%**。固定 cell 内配对环块区间为 **{primary['paired_within_cell_block_interval95']}**。", "",
        f"只有 **{improving[0]['cell']}** 改善，cell 内相对差为 **{100*improving[0]['relative_difference']:.8f}%**；其余 21 个 cell 按 1e−10 容差持平。改进出现于该 cell 的六个 learner，不能把六个重复结构当作六个独立市场证据。", "",
        "该 CI 固定了 22 个市场×年份×期限 cell，只在各 cell 内以 2h 环块重采样起点。它不重采样市场、不捕捉跨 cell 相关性，不能证明跨市场可迁移增益或未来分布变化下的改善。60% cell 改善条件失败，本轮维持 STOP。", "",
        "计数应区分：63 份校准结果至少一坐标选择非零控制系数，不等于 63 个实际预测更新。主方法只有 14 个非零 gate，只有 6 个作业的已发出 gate 路径不同于 same-joint raw，且全部位于同一个 cell。", "",
        "## 训练与 epoch 0 的解释", "", "46/132 选择了 epoch 0 初始化权重，这是验证集择优后的回退。全部 132 个作业均有 epoch 0–60 的 61 条完整验证历史，60 个训练 epoch，共计 7,920 epoch、116,640 个 optimizer steps，与每作业 60×ceil(ntrain/64) 完全一致。每份历史均存在实际数值变化。", "",
        "本次重放没有 optimizer、backward 或 fitting 调用，新增训练数与梯度步数均为零。训练完成数量来自已存完整收据、历史及源码 hash 对账；不把重放包装成重新观察了全部历史梯度执行。", "",
        "## 数值差异", "", "```json", json.dumps(maxima, ensure_ascii=False, indent=2), "```", "",
        "## 方法比较", "", "以下均为先平均 cell 内六个 learner 的 loss，再计算每 cell 比值并对 22 个 cell 等权平均；不是先混合六个 CDF 再评分。", "",
        "| 方法 | CRPS / F0 | 相对 same-joint raw | 改善 / 持平 / 变差 cell | price CRPS / F0 |", "|---|---:|---:|---:|---:|"]
    for row in method_results:
        r = row["vs_same_joint_raw"]
        report.append(f"| {row['method']} | {row['mean_cell_CRPS_ratio_F0']:.10f} | {r['mean_cell_relative_difference']:.10f} | {r['cells_better']} / {r['cells_tied']} / {r['cells_worse']} | {row['mean_cell_price_CRPS_ratio_F0']:.10f} |")
    report += ["", "## 边界与身份", "", "核验 checkpoint 内的 weights、sigma0、feature_mean、feature_sd 与 prepared 数组完全一致，train X 的中心与标准差符合标准化定义；各角色标签成熟边界与 ≤2023 日期限制通过。没有重读原始价格，所以不把这次复核表述为重新认证数据来源。", "",
        "Gaussian 预测的 coverage/width 在聚合层对账；其 normalized CRPS、price CRPS、均值 MSE 是全量独立重算。两个原子预测方法另全量重算了 coverage/width。", "",
        "JSON 收据列出全部 checkpoint/prediction/prepared hashes、冻结源码身份、bootstrap 约定与计数；不写逐点价格、预测因子或损失。", "",
        "冻结后的变化已显式保留：run_market.py 在 22 个 prepared panel 完成后加强了日期防护，改为先仅读日期并拒绝 2024+ 输入，再读价格。审计将该精确代码块逆换回旧顺序，整份文件 SHA256 严格恢复 MARKET_PREFIT 的 deca05dea6…，证明没有混入其他变化；其余冻结模块逐字节未变。修复和零重训记录来自 provenance/POST_START_AUDIT_FIXES.json。首次核验遇到该 hash 差异时先停止，第二轮数值检查通过但保留待解释状态，最后完成精确逆 patch 证明后才报告整体通过。", "",
        "复现命令：", "", "```bash", "python -B research/distribution_update_learning_20261019/audits/independent_market_replay.py", "```", ""]
    (HERE/"INDEPENDENT_MARKET_REPLAY_ZH.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"all_passed": all_passed, "checkpoints": len(manifests), "factor_values": replayed_factors,
                      "primary": primary, "improving_cells": improving, "json": str(outpath)}, ensure_ascii=False), flush=True)
    if not all_passed:
        raise SystemExit("Numerical replay completed; runner source identity remains unresolved")


if __name__ == "__main__":
    main()
