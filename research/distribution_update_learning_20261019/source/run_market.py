"""Prospectively specified development experiment. All private arrays stay local.

Usage: python run_market.py --data-dir .../development --local-dir /tmp/... \
       --public-dir .../results/market --workers 3
No prices later than 2023 are accepted. No network or GitHub calls are made.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
import torch

from scores import (crps, pair_terms, reflected_control, mixture_parts, quantiles,
                    lognormal_crps, atomic_crps)
from learning import block_gate, empirical_gate, circular_indices
from market_models import fit_global_mixture, fit_proposal, predict_factor, fit_garch
from online_baselines import BOACDFMixture, QuantileBOAG


MARKETS = ("EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI")
BACKBONES = ("MLP32_16", "GRU16")
SEEDS = (17, 41, 89)
METRICS = ("normalized_CRPS", "price_CRPS", "mean_price_squared_error", "coverage90", "width90", "alpha")
PRIMARY = "path_block_gate3_delta005"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def create_split(dates, year, h):
    years = pd.DatetimeIndex(dates).year.to_numpy()
    first = int(np.flatnonzero(years == year)[0])
    cal_end, cal_start = first-h-1, first-h-1-251
    val_end = cal_start-h-1
    val_start = val_end-125
    train_end = val_start-h-1
    train = np.arange(39, train_end+1)
    val, cal = np.arange(val_start, val_end+1), np.arange(cal_start, cal_end+1)
    test = np.flatnonzero((years[:len(years)-h] == year) & (years[h:] == year))
    if len(train) < 128 or len(val) != 126 or len(cal) != 252:
        raise ValueError("Prospective minimum/split rule failed")
    if not (train[-1]+h < val[0] and val[-1]+h < cal[0] and cal[-1]+h < test[0]):
        raise AssertionError("Immature label crosses a role boundary")
    return {"train": train, "validation": val, "calibration": cal, "test": test}


def prepare_cell(data_path, local_dir, market, year, h):
    started = time.time()
    date_only = pd.read_csv(data_path, usecols=["date"])
    dates = pd.to_datetime(date_only["date"])
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("This development runner refuses reserved-period prices")
    df = pd.read_csv(data_path)
    price = df["price"].to_numpy(float)
    if not dates.is_monotonic_increasing or dates.duplicated().any() or np.any(price <= 0) or not np.all(np.isfinite(price)):
        raise ValueError("Invalid source order, duplicates, or nonpositive/nonfinite price")
    roles = create_split(dates, year, h)
    train = roles["train"]
    logprice = np.log(price)
    returns = np.r_[0., np.diff(logprice)]
    training_label_end = int(train[-1]+h)
    floor = max(1e-8, .1 * np.median(np.abs(returns[1:training_label_end+1])))
    variance = np.full(len(price), np.mean(returns[1:21] ** 2))
    for i in range(21, len(price)):
        variance[i] = .94 * variance[i-1] + .06 * returns[i] ** 2
    daily_scale = np.maximum(np.sqrt(variance), floor)
    intervals = np.r_[1., np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(float)]
    features = np.stack([returns/daily_scale, np.log(daily_scale), np.log1p(intervals)], axis=1)
    X = np.stack([features[np.arange(i-19, i+1)] for i in range(39, len(price)-h)])
    origin = np.arange(39, len(price)-h)
    feature_mean = X[train-39].mean(axis=(0, 1))
    feature_sd = np.maximum(X[train-39].std(axis=(0, 1)), 1e-5)
    X = (X-feature_mean) / feature_sd
    origin_unit = daily_scale[origin] * np.sqrt(h)
    y = (logprice[origin+h]-logprice[origin]) / origin_unit
    path = (logprice[origin[:, None]+np.arange(1, h+1)] - logprice[origin[:, None]+np.arange(h)]) / origin_unit[:, None]
    np.testing.assert_allclose(path.sum(axis=1), y, atol=1e-10)
    lvmean, lvsd = np.log(daily_scale[train]).mean(), max(np.log(daily_scale[train]).std(), 1e-5)
    state = (np.log(daily_scale[origin])-lvmean) / lvsd
    basis = softmax(-.5 * ((state[:, None]-np.array([-1., 0., 1.]))/.75)**2, axis=-1)
    w, s, global_receipt = fit_global_mixture(y[train-39])
    gvar, glong, persistence, garch_receipt = fit_garch(returns, training_label_end)
    cumulative_gvar = h*glong+(gvar[origin]-glong)*(1-persistence**h)/(1-persistence)
    garch_sigma = np.sqrt(np.maximum(cumulative_gvar, 1e-20))/origin_unit
    fhs_atoms = np.quantile(y[train-39], (np.arange(101)+.5)/101)
    fhs_atoms = fhs_atoms-fhs_atoms.mean()
    arrays = {"weights": w, "sigma0": s, "feature_mean": feature_mean, "feature_sd": feature_sd,
              "history_logscale_mean": np.array(lvmean), "history_logscale_sd": np.array(lvsd),
              "fhs_atoms": fhs_atoms, "floor": np.array(floor)}
    for role, ids in roles.items():
        ix = ids-39
        arrays.update({f"{role}_X": X[ix], f"{role}_y": y[ix], f"{role}_basis": basis[ix],
                       f"{role}_path": path[ix], f"{role}_unit": origin_unit[ix],
                       f"{role}_origin_price": price[ids], f"{role}_target_price": price[ids+h],
                       f"{role}_garch_sigma": garch_sigma[ix], f"{role}_indices": ids,
                       f"{role}_dates": dates.iloc[ids].dt.strftime("%Y-%m-%d").to_numpy(dtype="U10")})
    cell = f"{market}_{year}_h{h}"
    path_out = local_dir/"prepared"/f"{cell}.npz"
    np.savez_compressed(path_out, **arrays)
    receipt = {"cell": cell, "market": market, "year": year, "horizon": h,
               "source_sha256": sha(data_path), "prepared_sha256": sha(path_out),
               "global_mixture_fit": global_receipt, "GARCH_fit": garch_receipt,
               "elapsed_seconds": time.time()-started,
               "roles": {role: {"count": len(ids), "first_origin": int(ids[0]), "last_origin": int(ids[-1]),
                                   "first_date": str(dates.iloc[ids[0]].date()), "last_date": str(dates.iloc[ids[-1]].date()),
                                   "last_label_index": int(ids[-1]+h), "last_label_date": str(dates.iloc[ids[-1]+h].date())}
                         for role, ids in roles.items()}}
    (local_dir/"prepared"/f"{cell}.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def gaussian_metrics(y, unit, price0, pricetrue, weights, sigmas, alpha=None):
    n = len(y)
    weights = np.broadcast_to(weights, (n, np.shape(weights)[-1]))
    sigmas = np.broadcast_to(sigmas, (n, np.shape(sigmas)[-1]))
    normscore = crps(y, weights, sigmas)
    log_sigma = unit[:, None]*sigmas
    log_location = np.broadcast_to(np.log(price0)[:, None], log_sigma.shape)
    pscore = lognormal_crps(pricetrue, log_location, log_sigma, weights)
    mean = np.sum(weights*np.exp(log_location+.5*log_sigma**2), axis=-1)
    q = quantiles(np.array([.05, .95]), weights, sigmas)
    pq = price0[:, None] * np.exp(unit[:, None]*q)
    coverage = ((pricetrue >= pq[:, 0]) & (pricetrue <= pq[:, 1])).astype(float)
    return np.stack([normscore, pscore, (mean-pricetrue)**2, coverage, pq[:, 1]-pq[:, 0],
                     np.full(n, np.nan) if alpha is None else np.broadcast_to(alpha, (n,))], axis=1)


def atomic_metrics(y, unit, price0, pricetrue, atoms):
    atoms = np.sort(np.broadcast_to(atoms, (len(y), np.shape(atoms)[-1])), axis=-1)
    price_atoms = price0[:, None]*np.exp(unit[:, None]*atoms)
    score = atomic_crps(y, atoms)
    pscore = atomic_crps(pricetrue, price_atoms)
    mean = price_atoms.mean(axis=-1)
    k = atoms.shape[-1]
    # Left-continuous generalized inverse of the issued finite CDF.
    low, high = int(np.ceil(.05*k))-1, int(np.ceil(.95*k))-1
    lo, hi = price_atoms[:, low], price_atoms[:, high]
    cov = ((pricetrue >= lo) & (pricetrue <= hi)).astype(float)
    return np.stack([score, pscore, (mean-pricetrue)**2, cov, hi-lo, np.full(len(y), np.nan)], axis=1)


def fit_job(local_root, cell, backbone, seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    local = Path(local_root)
    started = time.time()
    d = dict(np.load(local/"prepared"/f"{cell}.npz"))
    h = int(cell.rsplit("_h", 1)[1])
    w, s = d["weights"], d["sigma0"]
    model, fit_receipt = fit_proposal(d["train_X"], d["train_y"], d["validation_X"], d["validation_y"], w, s, backbone, seed)
    fcal, ftest = predict_factor(model, d["calibration_X"]), predict_factor(model, d["test_X"])
    ac, bc, _, _ = pair_terms(d["calibration_y"], w, s, s*fcal[:, None])
    at, bt, st0, st1 = pair_terms(d["test_y"], w, s, s*ftest[:, None])
    c, cap, spent = reflected_control(d["calibration_path"], w, s, fcal)
    cf, capfull, _ = reflected_control(d["calibration_path"], w, s, fcal, cap_fraction=1.)
    cu, _, _ = reflected_control(d["calibration_path"], w, s, fcal, capped=False)
    cn, _, _ = reflected_control(d["calibration_path"], w, s, fcal, no_prefix=True)
    if np.max(np.abs(cn), initial=0) > 1e-12:
        raise AssertionError("No-prefix negative control must vanish for symmetric scale experts")
    phi, phit = d["calibration_basis"], d["test_basis"]
    idx = circular_indices(len(ac), 2*h, 399, np.random.default_rng(stable_seed(cell)))
    scalar, scalar_test = np.ones((len(ac), 1)), np.ones((len(at), 1))
    alpha = {"F0": np.zeros(len(at)), "F1": np.ones(len(at)), "equal_mixture": np.full(len(at), .5)}
    alpha["empirical_scalar"] = scalar_test @ empirical_gate(ac, bc, scalar)
    alpha["empirical_gate3"] = phit @ empirical_gate(ac, bc, phi)
    gate_receipts = {}
    specifications = [
        ("raw_block_gate3", c, cap, phi, phit, 0., (0.,), None),
        ("raw_block_gate3_same_joint_budget", c, cap, phi, phit, 0., (0.,), (0., .5, 1.)),
        (PRIMARY, c, cap, phi, phit, .05, (0., .5, 1.), None),
        ("path_block_gate3_delta0", c, cap, phi, phit, 0., (0., .5, 1.), None),
        ("path_block_gate3_delta0025", c, cap, phi, phit, .025, (0., .5, 1.), None),
        ("path_block_gate3_delta010", c, cap, phi, phit, .10, (0., .5, 1.), None),
        ("path_block_scalar_delta005", c, cap, scalar, scalar_test, .05, (0., .5, 1.), None),
        ("path_fullcap_gate3_delta005", cf, capfull, phi, phit, .05, (0., .5, 1.), None),
        ("path_uncapped_gate3_delta005", cu, h*cap, phi, phit, .05, (0., .5, 1.), None),
        ("path_no_prefix_gate3_delta005", cn, cap, phi, phit, .05, (0., .5, 1.), None),
    ]
    for name, control, known_cap, basis, tbasis, budget, lambdas, joint in specifications:
        coefficients, receipt = block_gate(ac, bc, control, known_cap, basis, idx, budget, lambdas, joint)
        alpha[name] = tbasis @ coefficients
        gate_receipts[name] = receipt

    boa = BOACDFMixture()
    # This is training on already mature past calibration labels, not a claim
    # to reconstruct a delayed historical deployment during calibration.
    boa.replay_ab(ac, bc)
    alpha["BOA_CDF_frozen"] = np.full(len(at), boa.alpha)
    online = copy.deepcopy(boa)
    alpha_online = np.empty(len(at))
    tau = (np.arange(19)+.5)/19
    base_quantiles = quantiles(tau, w, s)
    qcal = np.stack([np.broadcast_to(base_quantiles, (len(ac), 19)), fcal[:, None]*base_quantiles], axis=-1)
    qt = np.stack([np.broadcast_to(base_quantiles, (len(at), 19)), ftest[:, None]*base_quantiles], axis=-1)
    quantile_model = QuantileBOAG(tau)
    quantile_model.replay_quantiles(qcal, d["calibration_y"])
    issued_atoms = np.empty((len(at), 19))
    observed, maturity_checks = 0, 0
    origins = d["test_indices"]
    for i, current_origin in enumerate(origins):
        while observed < i and origins[observed]+h <= current_origin:
            online.update_ab(at[observed], bt[observed], alpha_online[observed])
            quantile_model.update_quantiles(qt[observed], d["test_y"][observed], issued_atoms[observed])
            maturity_checks += 1
            observed += 1
        alpha_online[i] = online.alpha
        issued_atoms[i] = quantile_model.predict(qt[i])
    alpha["BOA_CDF_online"] = alpha_online
    metrics, names = [], []
    for name, prediction_alpha in alpha.items():
        prediction_alpha = np.clip(prediction_alpha, 0., 1.)
        mixed_w, mixed_s = mixture_parts(w, s, ftest, prediction_alpha)
        values = gaussian_metrics(d["test_y"], d["test_unit"], d["test_origin_price"], d["test_target_price"], mixed_w, mixed_s, prediction_alpha)
        # Independent-of-mixture-output quadratic identity at every test row.
        quadratic = st0+at*prediction_alpha**2-2*bt*prediction_alpha
        np.testing.assert_allclose(values[:, 0], quadratic, rtol=1e-9, atol=2e-10)
        names.append(name)
        metrics.append(values)
    names.append("quantile_BOAG_online")
    metrics.append(atomic_metrics(d["test_y"], d["test_unit"], d["test_origin_price"], d["test_target_price"], issued_atoms))
    names.append("FHS")
    metrics.append(atomic_metrics(d["test_y"], d["test_unit"], d["test_origin_price"], d["test_target_price"], d["fhs_atoms"]))
    for name, scale in (("EWMA_Normal", np.ones((len(at), 1))), ("GARCH_Normal", d["test_garch_sigma"][:, None])):
        names.append(name)
        metrics.append(gaussian_metrics(d["test_y"], d["test_unit"], d["test_origin_price"], d["test_target_price"], np.ones(1), scale))
    predictions = np.stack(metrics, axis=1)
    if not np.all(np.isfinite(predictions[:, :, :5])):
        raise ValueError("Nonfinite forecast score; do not silently discard a method")
    job = f"{cell}_{backbone}_s{seed}"
    checkpoint = local/"checkpoints"/f"{job}.pt"
    torch.save({"backbone": backbone, "state_dict": model.state_dict(), "seed": seed,
                "selected_epoch": fit_receipt["selected_epoch"], "weights": w, "sigma0": s,
                "feature_mean": d["feature_mean"], "feature_sd": d["feature_sd"]}, checkpoint)
    out = local/"predictions"/f"{job}.npz"
    np.savez_compressed(out, metrics=predictions, methods=np.array(names), metric_names=np.array(METRICS),
                        factors_calibration=fcal, factors_test=ftest, a_calibration=ac, b_calibration=bc,
                        control=c, cap=cap, spent=spent, a_test=at, b_test=bt, issued_quantile_atoms=issued_atoms,
                        test_indices=origins, rw_squared_error=(d["test_target_price"]-d["test_origin_price"])**2)
    receipt = {"job": job, "cell": cell, "backbone": backbone, "seed": seed,
               "fit": fit_receipt, "calibration_gates": gate_receipts,
               "n_test": len(at), "online_updates_with_verified_maturity": maturity_checks,
               "online_last_observed_origin": int(origins[observed-1]) if observed else None,
               "elapsed_seconds": time.time()-started, "checkpoint_sha256": sha(checkpoint),
               "predictions_sha256": sha(out), "mean_squared_error_RW": float(np.mean((d["test_target_price"]-d["test_origin_price"])**2)),
               "metrics": {name: {metric: (float(np.mean(predictions[:, j, k])) if k < 5 or np.isfinite(predictions[:, j, k]).any() else None)
                                     for k, metric in enumerate(METRICS)} for j, name in enumerate(names)}}
    (local/"receipts"/f"{job}.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return {"job": job, "selected_epoch": fit_receipt["selected_epoch"], "elapsed_s": round(time.time()-started, 2), "status": "completed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--mode", choices=("all", "prepare", "fit"), default="all")
    args = parser.parse_args()
    for child in ("prepared", "checkpoints", "predictions", "receipts"):
        (args.local_dir/child).mkdir(parents=True, exist_ok=True)
    args.public_dir.mkdir(parents=True, exist_ok=True)
    cells = [(m, y, h) for m in MARKETS for y in (2021, 2022, 2023) for h in (5, 20) if not (m == "SHANGHAI" and y == 2021)]
    source_root = Path(__file__).resolve().parent
    if args.mode in ("all", "prepare"):
        freeze = {"timestamp_before_first_market_fit_unix": time.time(), "mode": "fixed_development",
                  "source_sha256": {p.name: sha(p) for p in sorted(source_root.glob("*.py"))},
                  "protocol_sha256": sha(source_root.parent/"provenance"/"PROTOCOL.json"),
                  "addendum_sha256": sha(source_root.parent/"provenance"/"DESIGN_ADDENDUM_PRE_RUN.json"),
                  "cells": cells, "backbones": BACKBONES, "seeds": SEEDS,
                  "new_confirmation_prices_opened": False,
                  "quantile_baseline": "19 equal atoms at tau=(j+0.5)/19; exact atomic CRPS, unsmoothed BOAG adaptation"}
        (args.public_dir/"MARKET_PREFIT.json").write_text(json.dumps(freeze, indent=2))
        preparations = []
        for market, year, h in cells:
            receipt = prepare_cell(args.data_dir/f"{market}_development.csv", args.local_dir, market, year, h)
            preparations.append(receipt)
            print(json.dumps({"prepared": receipt["cell"], "seconds": round(receipt["elapsed_seconds"], 2)}), flush=True)
        (args.public_dir/"PREPARATION_RECEIPTS.json").write_text(json.dumps(preparations, indent=2))
    if args.mode in ("all", "fit"):
        jobs = [(str(args.local_dir), f"{m}_{y}_h{h}", backbone, seed) for m, y, h in cells for backbone in BACKBONES for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fit_job, *job): job for job in jobs}
            completed = []
            for future in as_completed(futures):
                receipt = future.result()
                completed.append(receipt)
                print(json.dumps({"completed_jobs": len(completed), "total_jobs": len(jobs), **receipt}), flush=True)
        (args.public_dir/"JOB_COMPLETION.json").write_text(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
