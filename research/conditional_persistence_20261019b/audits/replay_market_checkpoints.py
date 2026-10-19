"""Independent inference-only replay of the frozen 312 + 198 market jobs.

No training-source module is imported. Standard PyTorch layers are assembled
from the documented architecture; probability transforms and the Markov
count-polynomial recursion are independent implementations. No CRPS, price
score, or model performance is recomputed or compared.

Role disclosure: the reviewer authored data preparation and the GAS runner's
exact-split input guards. Models, filters, optimizers and prediction code were
authored by other researchers. Preparation validation is not independent of
its author; the model/checkpoint replay is cross-review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STUDY = Path(__file__).resolve().parents[1]
MARKETS = ("EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI")
OLD_PANELS = tuple(f"{m}_{y}" for m in MARKETS for y in (2021, 2022, 2023)
                   if (m, y) != ("SHANGHAI", 2021))
PANELS = OLD_PANELS + ("SHENZHEN_PLOS_2020", "SHENZHEN_PLOS_2021")
ARMS = {
    "GRU16": ("conditional_joint005", "constant_joint005",
              "conditional_endpoint", "constant_endpoint",
              "scale_capacity_joint005", "independent_horizon_endpoint"),
    "MLP16": ("conditional_joint005", "constant_joint005"),
}
SEEDS, ROLES, HORIZONS = (17, 41, 89), ("train", "validation", "test"), (1, 5, 20)
SOURCE_SHA = {
    "EUA_LONG": "142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f",
    "GUANGZHOU": "8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051",
    "HUBEI": "04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797",
    "SHANGHAI": "425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1",
    "SHENZHEN_PLOS": "90c49501afa9beb7370c4409084a28729daa4904a7e7e5de310d0f6bd7a3ba10",
}
CANDIDATE_IDENTITY = {
    "protocol_sha256": "f2ac9d5aac5a8bbc7d6f7a785e943e307b903d2fc3ab56cc8484b9e1375adc9d",
    "source_sha256": {
        "source/run_persistence_market.py": "6aadf66adef6cdd7f193000ad09c472c1b4069a86ccad997d90fa63bb7fd480a",
        "source/persistence_market.py": "47de57b450e1e0e89ab7b4b8c061e681f9e17dca270a07807af9cc8969db0016",
        "source/persistence.py": "4c7dfa7a941089d9b21f54508dbff3b7446a8ec9af90a524fcd65d12ca5a92eb",
        "source/mixture_metrics.py": "6c8e69fad2bd07cecdfbbd67a2189b79f2a85678102dc1f167bc97968bee241d",
        "data_sources/prepare_data.py": "9b71d032bd6c401622b9f0b0e4d23b1ee9bf303d58540b90564d9506ab49f164",
    },
}
GAS_FILES = {
    "source/gas_filter.py": "57c23a002e221565c5e67766012659646c3e1a1e0524cb937e884c0d5c309de8",
    "source/mixture_metrics.py": "6c8e69fad2bd07cecdfbbd67a2189b79f2a85678102dc1f167bc97968bee241d",
    "source/run_gas_benchmark.py": "884f20b9ff2115f7ba9ee35bbc05f1380f46332db88a71f359dcc6f8372b93d3",
    "data_sources/prepare_data.py": "9b71d032bd6c401622b9f0b0e4d23b1ee9bf303d58540b90564d9506ab49f164",
    "provenance/PROTOCOL.json": "88c72e1781084473a426f0276869c63d78ed5076b088e2291a1c97a47dff77b8",
    "provenance/GAS_IMPLEMENTATION_PRE_FIT.json": "d5d8fac5be8daf05359ef6d51de7ee5d3c49a4f4a6daee1ef5d6d84f5b862bd8",
}
GAS_FINGERPRINT = "8bc1e84129b65fc30db010839523a448e605fae07581fb224a1b7b5e4c2dbf23"
ATOL = RTOL = 2e-11


def require(condition, label):
    if not bool(condition):
        raise AssertionError(label)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def compare(actual, expected, label, errors, *, exact=False):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, label + ": shape mismatch")
    if exact or a.dtype.kind in "USbiu":
        require(np.array_equal(a, b), label + ": exact identity mismatch")
        return
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)), label + ": nonfinite")
    difference = float(np.max(np.abs(a - b), initial=0.0))
    errors[label] = difference
    require(np.all(np.abs(a - b) <= ATOL + RTOL * np.abs(b)),
            label + ": numerical mismatch; max absolute error=" + repr(difference))


def market_of(panel):
    return panel.rsplit("_", 1)[0]


def check_dates(dates, market, label):
    dates = np.asarray(dates).astype("U10")
    require(dates.size > 0, label + ": no dates")
    cutoff = "2022-01-01" if market == "SHENZHEN_PLOS" else "2024-01-01"
    require(np.all(dates < cutoff), label + ": reserved dates rejected before numeric access")
    return dates


def load_base(directory, panel):
    path = directory / (panel + ".npz")
    receipt, digest = read_json(path.with_suffix(".json")), sha(path)
    market, year = market_of(panel), panel.rsplit("_", 1)[1]
    require(receipt["cell"] == panel and receipt["prepared_sha256"] == digest,
            panel + ": producer identity mismatch")
    require(receipt["source"]["sha256"] == SOURCE_SHA[market], panel + ": source identity mismatch")
    with np.load(path, allow_pickle=False) as z:
        dates = check_dates(z["all_dates"], market, panel)  # Before numeric arrays.
        require(np.all(dates[1:] > dates[:-1]), panel + ": dates not strictly ordered")
        require(str(z["source_sha256"].item()) == SOURCE_SHA[market], panel + ": NPZ source mismatch")
        in_year = np.char.startswith(dates, year + "-")
        first_test = int(np.flatnonzero(in_year)[0])
        expected = {
            "train": np.arange(39, first_test - 166),
            "validation": np.arange(first_test - 146, first_test - 20),
            "test": np.flatnonzero(in_year[:-20] & in_year[20:]),
        }
        arrays, errors = {"all_dates": dates}, {}
        for role in ROLES:
            ids = z[role + "_indices"]
            compare(ids, expected[role], role + "_split", errors, exact=True)
            compare(z[role + "_dates"], dates[ids], role + "_dates", errors, exact=True)
            targets = ids[:, None] + np.array(HORIZONS)[None, :]
            compare(z[role + "_target_indices"], targets, role + "_targets", errors, exact=True)
            compare(z[role + "_target_dates"], dates[targets], role + "_target_dates", errors, exact=True)
            require(receipt["roles"][role]["count"] == len(ids), role + ": producer count mismatch")
            for suffix in ("indices", "dates", "target_indices", "target_dates", "X", "unit"):
                arrays[role + "_" + suffix] = z[role + "_" + suffix].copy()
        arrays.update({key: z[key].copy() for key in
                       ("train_label_end", "feature_mean", "feature_sd", "floor")})
        require(len(expected["train"]) >= 128, panel + ": insufficient fixed training origins")
        require(len(expected["validation"]) == 126, panel + ": changed validation budget")
        require(int(arrays["train_label_end"]) == expected["train"][-1] + 20 < expected["validation"][0],
                panel + ": training maturity failure")
        require(expected["validation"][-1] + 20 < expected["test"][0], panel + ": validation maturity failure")
    return {"arrays": arrays, "receipt": receipt, "sha256": digest,
            "receipt_sha256": sha(path.with_suffix(".json"))}


def independent_raw(state, x, backbone, *, gas=False):
    """Standard-layer assembly; no producer class or forward function used."""
    x = np.asarray(x, dtype=np.float32)
    prefix = "gru." if gas else "encoder."
    if gas or backbone == "GRU16":
        encoder = nn.GRU(1 if gas else 3, 16, batch_first=True)
        encoder.load_state_dict({k[len(prefix):]: value for k, value in state.items()
                                 if k.startswith(prefix)}, strict=True)
        encoder.eval()
    else:
        encoder = None
    batch, output = (256 if gas else 128), []
    with torch.no_grad():
        for first in range(0, len(x), batch):
            part = torch.from_numpy(x[first:first + batch])
            if encoder is not None:
                sequence, hidden = encoder(part)
                encoded = hidden[-1] if gas else sequence[:, -1, :]
            else:
                encoded = torch.tanh(F.linear(part.flatten(start_dim=1),
                                             state["encoder.1.weight"], state["encoder.1.bias"]))
                encoded = torch.tanh(F.linear(encoded, state["encoder.3.weight"],
                                             state["encoder.3.bias"]))
            raw = F.linear(encoded, state["head.weight"], state["head.bias"])
            output.append(raw.cpu().numpy().astype(np.float64))
    return np.concatenate(output, axis=0)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def candidate_parameters(raw, reference, arm):
    scale_index = raw[:, 0]
    if arm == "scale_capacity_joint005":
        scale_index = scale_index + 0.5 * np.tanh(raw[:, 1])
    f = reference["f0"] * (0.5 + sigmoid(scale_index))
    mode = "joint005" if arm.endswith("joint005") else "endpoint"
    rho0 = reference["rho_" + mode]
    base_logit = math.log(rho0) - math.log1p(-rho0)
    if arm == "independent_horizon_endpoint":
        rho = sigmoid(base_logit + raw[:, 1:3])
    elif arm.startswith("conditional_"):
        value = sigmoid(base_logit + raw[:, 1])
        rho = np.repeat(value[:, None], 2, axis=1)
    else:
        rho = np.full((len(raw), 2), rho0)
    return {"f": f, "rho_h5": rho[:, 0], "rho_h20": rho[:, 1]}


def independent_dp(rho, pi, horizons=HORIZONS):
    """Explicitly sum nine state transitions with integer polynomial shifts.

    This does not reuse the producer's persistence-plus-redraw recursion.
    The exponent is n_middle + 5*n_high, not a floating-point grid bin.
    """
    rho, pi, shift = np.asarray(rho, dtype=float), np.asarray(pi, dtype=float), (0, 1, 5)
    mass = np.zeros((len(rho), 3, 6), dtype=np.float64)
    for state in range(3):
        mass[:, state, shift[state]] = pi[state]
    result = {}
    for h in range(1, max(horizons) + 1):
        if h in horizons:
            result[h] = mass.sum(axis=1)
        if h == max(horizons):
            break
        next_mass = np.zeros((len(rho), 3, 5 * (h + 1) + 1), dtype=np.float64)
        for previous in range(3):
            for current in range(3):
                probability = (1.0 - rho) * pi[current] + rho * (previous == current)
                offset = shift[current]
                next_mass[:, current, offset:offset + mass.shape[2]] += (
                    mass[:, previous, :] * probability[:, None])
        mass = next_mass
    return result


def dp_self_check():
    """Full 243-path enumeration at h5, including both persistence endpoints."""
    pi, rho, shift = np.array([0.21, 0.53, 0.26]), np.array([0.0, 0.31, 0.91, 1.0]), (0, 1, 5)
    expected = np.zeros((len(rho), 26))
    for states in itertools.product(range(3), repeat=5):
        probability = np.full(len(rho), pi[states[0]])
        for previous, current in zip(states[:-1], states[1:]):
            probability *= (1.0 - rho) * pi[current] + rho * (previous == current)
        expected[:, sum(shift[k] for k in states)] += probability
    errors, actual = {}, independent_dp(rho, pi, (5,))[5]
    compare(actual, expected, "dp_vs_243_path_enumeration", errors)
    compare(actual.sum(axis=1), np.ones(len(rho)), "dp_total_mass", errors)
    return errors


def optimizer_steps(optimizer, expected, label):
    state = optimizer["state"]
    if expected == 0:
        require(len(state) == 0, label + ": epoch-zero optimizer must be empty")
    else:
        require(len(state) > 0, label + ": missing optimizer state")
        require(all(int(v["step"].item()) == expected for v in state.values()),
                label + ": optimizer steps differ from fixed budget")


def check_candidate_history(fit, ntrain):
    per_epoch, history = math.ceil(ntrain / 64), fit["history"]
    require([r["epoch"] for r in history] == list(range(41)), "Candidate epoch history incomplete")
    require([r["optimizer_steps"] for r in history] == [i * per_epoch for i in range(41)],
            "Candidate epoch optimizer-step history mismatch")
    values = np.asarray([row["validation"]["endpoint_CRPS"] for row in history])
    require(np.all(np.isfinite(values)), "Candidate nonfinite validation selection history")
    selected = int(np.argmin(values))  # First minimum implements strict improvements.
    require(fit["selected_epoch"] == selected, "Candidate validation-only selected epoch mismatch")
    require(fit["selected_validation_CRPS"] == float(values[selected]), "Candidate selected validation receipt mismatch")
    require(fit["epochs_run"] == 40 and fit["optimizer_steps"] == 40 * per_epoch,
            "Candidate incomplete optimizer budget")
    require(fit["test_used_for_selection"] is False and fit["training_complete_before_validation_selected_state_reload"] is True,
            "Candidate selection-scope receipt mismatch")
    return selected, 40 * per_epoch, selected * per_epoch


def replay_candidate(panel, backbone, arm, seed, private, public, base, code_sha, cache):
    job = f"{panel}__{backbone}__{arm}__s{seed}"
    receipt_path = public / "fits" / (job + ".json")
    receipt = read_json(receipt_path)
    checkpoint_path = private / "checkpoints" / (job + ".pt")
    forecast_path = private / "forecasts" / (job + ".npz")
    reference_path = public / "references" / (panel + ".json")
    artifacts = {
        "receipt_sha256": sha(receipt_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "forecast_sha256": sha(forecast_path),
        "prepared_sha256": base["sha256"],
        "producer_receipt_sha256": base["receipt_sha256"],
        "reference_receipt_sha256": sha(reference_path),
    }
    require(receipt["job"] == job and receipt["panel"] == panel, job + ": job identity mismatch")
    require(receipt["identity"] == CANDIDATE_IDENTITY, job + ": frozen source identity mismatch")
    expected_artifacts = {"checkpoints/" + job + ".pt": artifacts["checkpoint_sha256"],
                          "forecasts/" + job + ".npz": artifacts["forecast_sha256"]}
    require(receipt["private_artifacts"] == expected_artifacts, job + ": artifact hash mismatch")
    prep = receipt["prepared"]
    require(prep["panel"] == panel and prep["prepared_sha256"] == base["sha256"]
            and prep["producer_receipt_sha256"] == base["receipt_sha256"]
            and prep["source_sha256"] == SOURCE_SHA[market_of(panel)]
            and prep["producer"] == base["receipt"], job + ": prepared linkage mismatch")
    cache_path, cache_key = cache / (job + ".json"), {"audit_code_sha256": code_sha, **artifacts}
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached["cache_key"] == cache_key:
            return cached["result"], True
    d = base["arrays"]
    c = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require((c["panel"], c["backbone"], c["arm"], c["seed"]) == (panel, backbone, arm, seed),
            job + ": checkpoint job mismatch")
    require(c["identity"] == CANDIDATE_IDENTITY and c["prepared_sha256"] == base["sha256"]
            and c["source_sha256"] == SOURCE_SHA[market_of(panel)], job + ": checkpoint linkage mismatch")
    require(c["fit_receipt"] == receipt["fit"], job + ": checkpoint/public fit receipt mismatch")
    reference_receipt = read_json(reference_path)
    require(reference_receipt["panel"] == panel
            and reference_receipt["prepared"]["prepared_sha256"] == base["sha256"]
            and reference_receipt["prepared"]["producer_receipt_sha256"] == base["receipt_sha256"]
            and reference_receipt["prepared"]["source_sha256"] == SOURCE_SHA[market_of(panel)]
            and reference_receipt["statistical_fits"] == 3
            and reference_receipt["identity"] == CANDIDATE_IDENTITY
            and c["reference"] == reference_receipt["reference"], job + ": fixed reference mismatch")
    errors = {}
    for key in ("train_indices", "validation_indices", "feature_mean", "feature_sd", "floor", "train_label_end"):
        compare(c[key], d[key], "checkpoint_" + key, errors)
    ntrain, ntest = len(d["train_indices"]), len(d["test_indices"])
    require(receipt["fit"]["train_origins"] == ntrain and receipt["fit"]["validation_origins"] == 126,
            job + ": fit role count mismatch")
    selected, completed_steps, selected_steps = check_candidate_history(receipt["fit"], ntrain)
    require(c["selected_epoch"] == selected and c["completed_optimizer_steps"] == completed_steps,
            job + ": checkpoint epoch/budget mismatch")
    optimizer_steps(c["final_optimizer_state_dict"], completed_steps, job + ": final")
    optimizer_steps(c["selected_optimizer_state_dict"], selected_steps, job + ": selected")
    require(receipt["market_neural_fits"] == 1 and receipt["test_parameter_updates"] == 0,
            job + ": fitting/deployment count mismatch")
    state, reference = c["model_state_dict"], c["reference"]
    require(all(torch.isfinite(value).all().item() for value in state.values()), job + ": nonfinite model state")
    parameters = {key: value for key, value in state.items()
                  if key.startswith("encoder.") or key.startswith("head.")}
    require(sum(value.numel() for value in parameters.values()) == receipt["fit"]["parameter_count"],
            job + ": parameter budget mismatch")
    pi = np.asarray(reference["pi"], dtype=float)
    rho0 = reference["rho_joint005" if arm.endswith("joint005") else "rho_endpoint"]
    q_mean = float(pi @ np.array([0.25, 1.0, 4.0]))
    for key, value in {"pi": pi, "f0": reference["f0"], "base_rho": rho0,
                       "base_logit": math.log(rho0) - math.log1p(-rho0),
                       "state_sigma": np.sqrt(np.array([0.25, 1.0, 4.0]) / q_mean)}.items():
        compare(state[key].numpy(), value, "state_" + key, errors)
    skeleton = {}
    for h in HORIZONS:
        grid = np.arange(5 * h + 1, dtype=float)
        skeleton[h] = np.sqrt((0.25 * h + 0.75 * grid) / (h * q_mean))
        compare(state[f"sigma_{h}"].numpy(), skeleton[h], f"state_sigma_{h}", errors)
        pair = math.sqrt(2 / math.pi) * np.sqrt(skeleton[h][:, None]**2 + skeleton[h][None, :]**2)
        compare(state[f"pair_{h}"].numpy(), pair, f"state_pair_{h}", errors)
    with np.load(forecast_path, allow_pickle=False) as z:
        # Check every stored role's dates before requesting forecast numbers.
        for role in ROLES:
            check_dates(z[role + "_dates"], market_of(panel), job + ": " + role)
            if role != "train":
                check_dates(z[role + "_target_dates"], market_of(panel), job + ": " + role + " targets")
            compare(z[role + "_indices"], d[role + "_indices"], role + "_origin_identity", errors, exact=True)
            compare(z[role + "_dates"], d[role + "_dates"], role + "_date_identity", errors, exact=True)
        for role in ("validation", "test"):
            for suffix in ("target_indices", "target_dates", "unit"):
                compare(z[role + "_" + suffix], d[role + "_" + suffix], role + "_" + suffix, errors)
        for role in ("train", "test"):
            raw = independent_raw(state, d[role + "_X"], backbone)
            p = candidate_parameters(raw, reference, arm)
            for key, value in p.items():
                compare(z[role + "_" + key], value, role + "_" + key, errors)
            if role == "train":
                continue
            if arm == "independent_horizon_endpoint":
                d5 = independent_dp(p["rho_h5"], pi, (1, 5))
                d20 = independent_dp(p["rho_h20"], pi, (20,))
                distributions = {1: d5[1], 5: d5[5], 20: d20[20]}
            else:
                distributions = independent_dp(p["rho_h5"], pi)
            for h, weights in distributions.items():
                compare(z[f"test_weights_h{h}"], weights, f"test_weights_h{h}", errors)
                compare(z[f"test_sigmas_h{h}"], p["f"][:, None] * skeleton[h][None, :],
                        f"test_sigmas_h{h}", errors)
                compare(z[f"test_means_h{h}"], np.zeros_like(weights), f"test_means_h{h}", errors)
                compare(weights.sum(axis=1), np.ones(ntest), f"test_mass_h{h}", errors)
                require(np.all(weights >= 0), job + ": negative mixture mass")
    result = {"job": job, "kind": "candidate", "status": "PASS",
              "hashes": artifacts, "source_sha256": SOURCE_SHA[market_of(panel)],
              "train_parameter_origins": ntrain, "test_distribution_origins": ntest,
              "test_horizons": list(HORIZONS), "selected_epoch_identity_checked": True,
              "forty_epoch_budget_checked": True, "optimizer_steps_checked": completed_steps,
              "max_absolute_errors": errors, "scores_recomputed": 0, "fits_performed": 0}
    write_json(cache_path, {"cache_key": cache_key, "result": result})
    return result, False


def load_gas_prepared(directory, cell, base):
    path = directory / "prepared" / (cell + ".npz")
    meta, digest = read_json(path.with_suffix(".json")), sha(path)
    panel, h = cell.rsplit("_h", 1)
    h = int(h)
    require(meta["source_fingerprint"] == GAS_FINGERPRINT
            and meta["prepared_sha256"] == digest
            and meta["base_prepared_sha256"] == base["sha256"], cell + ": prepared chain mismatch")
    arrays, errors, b = {}, {}, base["arrays"]
    with np.load(path, allow_pickle=False) as z:
        for role in ROLES:
            check_dates(z[role + "_dates"], market_of(panel), cell + ": " + role)
            check_dates(z[role + "_target_dates"], market_of(panel), cell + ": " + role + " targets")
        require(str(z["source_sha256"].item()) == SOURCE_SHA[market_of(panel)],
                cell + ": derived source mismatch")
        require(int(z["horizon"]) == h and int(z["train_label_end"]) == int(b["train_label_end"]),
                cell + ": horizon or training maturity mismatch")
        k = HORIZONS.index(h)
        for role in ROLES:
            for suffix in ("indices", "dates"):
                compare(z[role + "_" + suffix], b[role + "_" + suffix],
                        role + "_" + suffix, errors, exact=True)
                arrays[role + "_" + suffix] = z[role + "_" + suffix].copy()
            for suffix in ("target_indices", "target_dates", "unit"):
                compare(z[role + "_" + suffix], b[role + "_" + suffix][:, k],
                        role + "_" + suffix, errors)
                arrays[role + "_" + suffix] = z[role + "_" + suffix].copy()
        for name in ("global", "ewma", "gas"):
            for suffix in ("X", "loc", "scale"):
                key = "test_" + suffix + "_" + name
                arrays[key] = z[key].copy()
    return {"arrays": arrays, "meta": meta, "sha256": digest,
            "receipt_sha256": sha(path.with_suffix(".json")),
            "base_prepared_sha256": base["sha256"]}


def gas_distribution(raw, loc, scale, unit):
    w0 = np.array([0.45, 0.4, 0.15])
    sd0 = np.array([0.5, 1.0, 2.0])
    sd0 = sd0 / np.sqrt(w0 @ (sd0**2))
    logits = raw[:, :3] + np.log(w0)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(shifted)
    weights = e / e.sum(axis=1, keepdims=True)
    means = (loc[:, None] + scale[:, None] * raw[:, 3:6]) / unit[:, None]
    z = raw[:, 6:9] + np.log(np.expm1(sd0 - 1e-4))
    softplus = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    sigmas = scale[:, None] * (softplus + 1e-4) / unit[:, None]
    return weights, means, sigmas


def replay_gas(cell, name, seed, private, prepared, code_sha, cache):
    family, job = name + "_gru_gmm", f"{cell}__{name}_gru_gmm__s{seed}"
    prediction_path = private / "predictions" / cell / family / f"seed_{seed}.npz"
    checkpoint_path = private / "checkpoints" / cell / family / f"seed_{seed}.pt"
    receipt_path = prediction_path.with_suffix(".json")
    receipt = read_json(receipt_path)
    panel, h = cell.rsplit("_h", 1)
    h = int(h)
    artifacts = {"receipt_sha256": sha(receipt_path), "checkpoint_sha256": sha(checkpoint_path),
                 "prediction_sha256": sha(prediction_path), "prepared_sha256": prepared["sha256"],
                 "producer_receipt_sha256": prepared["receipt_sha256"],
                 "base_prepared_sha256": prepared["base_prepared_sha256"]}
    require((receipt["cell"], receipt["panel"], receipt["horizon"], receipt["family"], receipt["seed"])
            == (cell, panel, h, family, seed), job + ": job identity mismatch")
    require(receipt["source_fingerprint"] == GAS_FINGERPRINT
            and receipt["prepared_sha256"] == prepared["sha256"]
            and receipt["prediction_sha256"] == artifacts["prediction_sha256"]
            and receipt["checkpoint_sha256"] == artifacts["checkpoint_sha256"],
            job + ": frozen source/artifact identity mismatch")
    cache_path, cache_key = cache / (job + ".json"), {"audit_code_sha256": code_sha, **artifacts}
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached["cache_key"] == cache_key:
            return cached["result"], True
    d, meta = prepared["arrays"], prepared["meta"]
    c = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(c["source_fingerprint"] == GAS_FINGERPRINT and c["prepared_sha256"] == prepared["sha256"]
            and c["seed"] == seed and c["normalizer"] == meta["normalizers"][name]
            and c["history"] == receipt["history"], job + ": checkpoint linkage mismatch")
    ntrain, ntest = len(d["train_indices"]), len(d["test_indices"])
    history = receipt["history"]
    require([r["epoch"] for r in history] == list(range(41)), job + ": incomplete epoch history")
    values = np.asarray([row["validation_normalized_CRPS"] for row in history])
    require(np.all(np.isfinite(values)), job + ": nonfinite selection history")
    selected = int(np.argmin(values))
    require(receipt["selected_epoch"] == selected == c["selected_epoch"]
            and receipt["best_validation_normalized_CRPS"] == float(values[selected]),
            job + ": earliest validation-only selection mismatch")
    completed_steps = 40 * math.ceil(ntrain / 64)
    require(receipt["neural_fit"] is True and receipt["epochs_executed"] == 40
            and receipt["gradient_updates"] == completed_steps
            and receipt["test_origins"] == ntest, job + ": optimizer/role budget mismatch")
    optimizer_steps(c["final_optimizer_state"], completed_steps, job + ": final")
    state = c["model_state"]
    require(all(torch.isfinite(value).all().item() for value in state.values()), job + ": nonfinite selected state")
    require(sum(value.numel() for value in state.values()) == receipt["neural_parameters"] == 1065,
            job + ": GRU/GMM parameter budget mismatch")
    raw = independent_raw(state, d["test_X_" + name], "GRU16", gas=True)
    weights, means, sigmas = gas_distribution(raw, d["test_loc_" + name],
                                              d["test_scale_" + name], d["test_unit"])
    errors = {}
    with np.load(prediction_path, allow_pickle=False) as z:
        check_dates(z["date"], market_of(panel), job + ": prediction dates")
        check_dates(z["target_date"], market_of(panel), job + ": prediction target dates")
        for key, suffix in {"origin": "indices", "date": "dates", "target_index": "target_indices",
                            "target_date": "target_dates", "unit": "unit"}.items():
            compare(z[key], d["test_" + suffix], "prediction_" + key, errors)
        for key, value in {"weights": weights, "means": means, "sigmas": sigmas}.items():
            compare(z[key], value, "test_" + key, errors)
    compare(weights.sum(axis=1), np.ones(ntest), "test_total_mass", errors)
    require(np.all(weights > 0) and np.all(sigmas > 0), job + ": invalid GMM")
    result = {"job": job, "kind": "gas", "status": "PASS", "hashes": artifacts,
              "source_sha256": SOURCE_SHA[market_of(panel)], "horizon": h,
              "test_distribution_origins": ntest, "selected_epoch_identity_checked": True,
              "forty_epoch_budget_checked": True, "optimizer_steps_checked": completed_steps,
              "max_absolute_errors": errors, "scores_recomputed": 0, "fits_performed": 0}
    write_json(cache_path, {"cache_key": cache_key, "result": result})
    return result, False


def completion_inventory(public, expected_jobs):
    files = sorted(public.glob("COMPLETE_*.json"))
    require(len(files) <= 3, "Unexpected number of candidate completion manifests")
    all_jobs, all_panels, receipts = [], [], []
    for path in files:
        manifest = read_json(path)
        require(manifest["identity"] == CANDIDATE_IDENTITY
                and manifest["no_test_parameter_updates"] is True,
                path.name + ": completion identity or deployment mismatch")
        panels = manifest["panels"]
        jobs = manifest["jobs"]
        require(len(set(panels)) == len(panels) and set(panels) <= set(PANELS),
                path.name + ": completion panel identity mismatch")
        expected = {f"{p}__{b}__{a}__s{s}" for p in panels
                    for b, arms in ARMS.items() for a in arms for s in SEEDS}
        require(len(jobs) == len(expected) and {r["job"] for r in jobs} == expected,
                path.name + ": completion job inventory mismatch")
        total_steps = 0
        for row in jobs:
            fit_path = public / "fits" / (row["job"] + ".json")
            require(sha(fit_path) == row["receipt_sha256"], path.name + ": completed receipt changed")
            total_steps += read_json(fit_path)["fit"]["optimizer_steps"]
        require(manifest["neural_fits_completed"] == len(jobs)
                and manifest["static_fits_completed"] == 3 * len(panels)
                and manifest["epochs_completed"] == 40 * len(jobs)
                and manifest["optimizer_steps"] == total_steps,
                path.name + ": completed budget mismatch")
        all_jobs.extend(r["job"] for r in jobs)
        all_panels.extend(panels)
        receipts.append({"name": path.name, "sha256": sha(path),
                         "panels": panels, "neural_fits": len(jobs),
                         "static_fits": 3 * len(panels)})
    require(len(set(all_jobs)) == len(all_jobs) and len(set(all_panels)) == len(all_panels),
            "Candidate shards overlap")
    complete = (len(files) == 3 and set(all_panels) == set(PANELS)
                and set(all_jobs) == expected_jobs)
    return {"verified_complete": complete, "manifest_count": len(files),
            "manifest_jobs": len(all_jobs), "receipts": receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prepared", type=Path, default=Path("/tmp/carbon-conditional-persistence-runs/prepared"))
    parser.add_argument("--candidate-private", type=Path, default=Path("/tmp/carbon-persistence-market-runs"))
    parser.add_argument("--candidate-public", type=Path, default=STUDY / "results/market/persistence")
    parser.add_argument("--gas-private", type=Path, default=Path("/tmp/carbon-gas-normalization-run"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/carbon-checkpoint-replay-cache"))
    parser.add_argument("--output", type=Path, default=STUDY / "audits/INDEPENDENT_CHECKPOINT_REPLAY.json")
    parser.add_argument("--partial", action="store_true", help="Allow incomplete inventory; never emits formal PASS")
    args = parser.parse_args()
    started, code_sha = time.monotonic(), sha(__file__)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_map = {**CANDIDATE_IDENTITY["source_sha256"], **GAS_FILES,
                  "provenance/MARKET_PROTOCOL.json": CANDIDATE_IDENTITY["protocol_sha256"]}
    for filename, expected in source_map.items():
        require(sha(STUDY / filename) == expected, "Current frozen source differs: " + filename)
    computed_gas = hashlib.sha256(json.dumps(GAS_FILES, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    require(computed_gas == GAS_FINGERPRINT, "Internal GAS fingerprint mismatch")
    self_check = dp_self_check()
    bases, gas_prepared, results, missing, failures = {}, {}, [], [], []
    cache_hits = 0
    candidate_jobs = [(panel, backbone, arm, seed) for panel in PANELS
                      for backbone, arms in ARMS.items() for arm in arms for seed in SEEDS]
    gas_jobs = [(f"{panel}_h{h}", name, seed) for panel in OLD_PANELS for h in (5, 20)
                for name in ("global", "ewma", "gas") for seed in SEEDS]
    require(len(candidate_jobs) == 312 and len(gas_jobs) == 198, "Expected budget changed")
    for panel, backbone, arm, seed in candidate_jobs:
        job = f"{panel}__{backbone}__{arm}__s{seed}"
        receipt_path = args.candidate_public / "fits" / (job + ".json")
        if not receipt_path.exists():
            missing.append({"kind": "candidate", "job": job})
            continue
        try:
            if panel not in bases:
                bases[panel] = load_base(args.base_prepared, panel)
            result, cached = replay_candidate(panel, backbone, arm, seed,
                                               args.candidate_private, args.candidate_public,
                                               bases[panel], code_sha, args.cache_dir)
            results.append(result)
            cache_hits += int(cached)
        except json.JSONDecodeError as exc:
            if args.partial:
                missing.append({"kind": "candidate", "job": job, "receipt_in_progress": True})
            else:
                failures.append({"kind": "candidate", "job": job, "error": type(exc).__name__})
        except Exception as exc:
            failures.append({"kind": "candidate", "job": job, "error": type(exc).__name__ + ": " + str(exc)})
        if len(results) and len(results) % 40 == 0:
            print(json.dumps({"technical_replays_completed": len(results), "failures": len(failures)}), flush=True)
    for cell, name, seed in gas_jobs:
        job = f"{cell}__{name}_gru_gmm__s{seed}"
        receipt_path = args.gas_private / "predictions" / cell / (name + "_gru_gmm") / f"seed_{seed}.json"
        if not receipt_path.exists():
            missing.append({"kind": "gas", "job": job})
            continue
        try:
            panel = cell.rsplit("_h", 1)[0]
            if panel not in bases:
                bases[panel] = load_base(args.base_prepared, panel)
            if cell not in gas_prepared:
                gas_prepared[cell] = load_gas_prepared(args.gas_private, cell, bases[panel])
            result, cached = replay_gas(cell, name, seed, args.gas_private,
                                        gas_prepared[cell], code_sha, args.cache_dir)
            results.append(result)
            cache_hits += int(cached)
        except json.JSONDecodeError as exc:
            if args.partial:
                missing.append({"kind": "gas", "job": job, "receipt_in_progress": True})
            else:
                failures.append({"kind": "gas", "job": job, "error": type(exc).__name__})
        except Exception as exc:
            failures.append({"kind": "gas", "job": job, "error": type(exc).__name__ + ": " + str(exc)})
        if len(results) and len(results) % 40 == 0:
            print(json.dumps({"technical_replays_completed": len(results), "failures": len(failures)}), flush=True)
    expected_candidate = {f"{p}__{b}__{a}__s{s}.pt" for p, b, a, s in candidate_jobs}
    actual_candidate = {p.name for p in (args.candidate_private / "checkpoints").glob("*.pt")}
    expected_gas = {f"{c}/{n}_gru_gmm/seed_{s}.pt" for c, n, s in gas_jobs}
    actual_gas = {p.relative_to(args.gas_private / "checkpoints").as_posix()
                  for p in (args.gas_private / "checkpoints").rglob("*.pt")}
    unexpected = {"candidate": sorted(actual_candidate - expected_candidate),
                  "gas": sorted(actual_gas - expected_gas)}
    if any(unexpected.values()):
        failures.append({"error": "Unexpected checkpoint inventory", "inventory": unexpected})
    try:
        completion = completion_inventory(args.candidate_public,
                                           {name[:-3] for name in expected_candidate})
    except Exception as exc:
        completion = {"verified_complete": False}
        failures.append({"error": "Completion inventory: " + type(exc).__name__ + ": " + str(exc)})
    count = {kind: sum(r["kind"] == kind for r in results) for kind in ("candidate", "gas")}
    max_errors = {}
    for row in results:
        for name, value in row["max_absolute_errors"].items():
            key = row["kind"] + "/" + name
            max_errors[key] = max(value, max_errors.get(key, 0.0))
    complete = (count == {"candidate": 312, "gas": 198} and not missing and not failures
                and completion["verified_complete"])
    status = ("FAIL" if failures else
              "PARTIAL_TECHNICAL_CHECK" if args.partial else
              "PASS_ALL_510_CHECKPOINT_REPLAYS" if complete else "INCOMPLETE")
    report = {
        "schema_version": 1, "status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent selected checkpoint forward/parameter/distribution replay; no market-score recomputation",
        "role_disclosure": "Reviewer implemented preparation and exact GAS input guards; model/normalizer/training code was independently authored",
        "audit_code_sha256": code_sha, "fixed_atol": ATOL, "fixed_rtol": RTOL,
        "expected_jobs": {"candidate": 312, "gas": 198}, "completed_replays": count,
        "full_inventory_verified": complete, "partial_mode": args.partial, "cache_hits": cache_hits,
        "candidate_completion_inventory": completion,
        "candidate_identity": CANDIDATE_IDENTITY, "gas_source_files": GAS_FILES,
        "gas_source_fingerprint": GAS_FINGERPRINT, "source_sha256": SOURCE_SHA,
        "base_prepared": {panel: {"prepared_sha256": b["sha256"],
                                  "producer_receipt_sha256": b["receipt_sha256"],
                                  "role_counts": {role: len(b["arrays"][role + "_indices"]) for role in ROLES}}
                          for panel, b in sorted(bases.items())},
        "independent_dp_self_check": self_check, "max_absolute_errors": max_errors,
        "fits_performed": 0, "market_scores_recomputed": 0, "reserved_2024plus_prices_read": False,
        "raw_csv_or_workbook_opened": False, "missing_jobs": missing, "failures": failures,
        "unexpected_checkpoints": unexpected, "jobs": results,
        "limitations": [
            "Replays selected states, not optimizer trajectories or full training reproducibility.",
            "Train replay covers f/rho; test replay covers full mixture parameters. Candidate validation distributions are not replayed here.",
            "Epoch selection is checked against saved validation history, without evaluating performance or revising selection.",
            "Standard PyTorch GRU kernels are shared software dependencies; architecture assembly, probability transforms and state recursion are independent.",
            "GAS static fits are linked by immutable preparation/checkpoint metadata; filter mathematics was separately reviewed before fitting.",
            "Root performs the separate per-origin CRPS and price-score audit.",
            "Technical PASS does not imply statistical improvement, economic value, or publication readiness.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(args.output, report)
    print(json.dumps({"status": status, "completed_replays": count, "failures": len(failures),
                      "missing": len(missing), "output": str(args.output)}, ensure_ascii=False), flush=True)
    if failures:
        print(json.dumps({"first_failures": failures[:5]}, ensure_ascii=False), flush=True)
    return 0 if (complete or args.partial) and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
