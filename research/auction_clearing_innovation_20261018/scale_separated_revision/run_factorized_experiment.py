"""Frozen, bounded CPU study. No confirmation files are opened by this runner."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from factorized_models import FactorizedDemandNetwork, PointNetwork, factorized_loss
from clearing_models import count_parameters


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Normalizer:
    def __init__(self, x, c, indices, columns):
        self.columns = columns
        xx = x[indices][:, :, columns]
        self.xmean = xx.mean((0, 1), keepdims=True)
        self.xstd = xx.std((0, 1), keepdims=True)
        self.xstd = np.where(self.xstd < .05, 1., self.xstd)
        self.cmean = c[indices].mean(0, keepdims=True)
        self.cstd = c[indices].std(0, keepdims=True)
        self.cstd = np.where(self.cstd < .05, 1., self.cstd)

    def transform(self, x, c):
        return (np.clip((x[:, :, self.columns]-self.xmean)/self.xstd, -12, 12).astype(np.float32),
                np.clip((c-self.cmean)/self.cstd, -12, 12).astype(np.float32))

    def state(self):
        return {k: v for k, v in vars(self).items()}


def construct(name, n_features, n_context, cfg):
    kw = dict(n_features=n_features, n_context=n_context, hidden=cfg["hidden_width"])
    if name in {"GRU_full", "MLP_full"}:
        return PointNetwork(**kw, kind="mlp" if name == "MLP_full" else "gru")
    modes = {"FDMC": "stochastic", "FDMC_no_supply": "no_supply", "FDMC_plugin": "plugin",
             "FDMC_unseparated": "unseparated", "FactorDirect": "direct"}
    return FactorizedDemandNetwork(**kw, mode=modes[name], components=cfg["components"],
                                   knots=cfg["price_knots"])


def selection_score(mean, indices, data, meta, horizons):
    good = data["completed"][indices].astype(bool)
    scores = []
    for horizon in horizons:
        mask = good & (meta.horizon.to_numpy()[indices] == horizon)
        if not mask.any():
            raise ValueError("Validation horizon has no observed price")
        ii = indices[mask]
        pred = data["anchor"][ii] + data["scale"][ii]*mean[mask]
        mse = np.mean((pred-data["price"][ii])**2)
        rw = np.mean((data["anchor"][ii]-data["price"][ii])**2)
        scores.append(float(mse/rw))
    return float(np.mean(scores))


def tensor_data(data, normalizer):
    x, context = normalizer.transform(data["x"], data["context"])
    answer = {"x": torch.from_numpy(x), "context": torch.from_numpy(context)}
    for key in ["base", "y", "completed", "bid_ratio", "has_bid", "median_z", "median_ratio",
                "has_median", "basis_z", "has_basis"]:
        answer[key] = torch.as_tensor(data[key], dtype=torch.float64 if key == "base" else torch.float32)
    return answer


def predict(model, tensors, indices, batch_size=256):
    model.eval()
    collected = {}
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start+batch_size]
            output = model(tensors["x"][batch], tensors["context"][batch], tensors["base"][batch])
            for key, value in output.items():
                arr = value.detach().cpu().numpy()
                if key != "mechanical_failure" and not np.isfinite(arr).all():
                    raise FloatingPointError(f"Nonfinite forecast {key}")
                collected.setdefault(key, []).append(arr)
    return {key: np.concatenate(value) for key, value in collected.items()}


def fit(name, seed, lr, epochs, train_indices, val_indices, data, meta, cfg, columns,
        private, public, fit_id):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    normalizer = Normalizer(data["x"], data["context"], train_indices, columns)
    tensors = tensor_data(data, normalizer)
    model = construct(name, len(columns), data["context"].shape[1], cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(seed)
    traces, best, best_epoch, best_state, stale = [], float("inf"), 0, None, 0
    started = time.perf_counter()
    updates = 0
    print(json.dumps({"event": "fit_start", "fit": fit_id, "training_rows": len(train_indices),
                      "validation_rows": 0 if val_indices is None else len(val_indices),
                      "max_epochs": epochs, "parameters": count_parameters(model)}), flush=True)
    for epoch in range(1, epochs+1):
        model.train()
        total, count, max_grad = 0., 0, 0.
        order = rng.permutation(train_indices)
        for start in range(0, len(order), cfg["batch_size"]):
            batch = order[start:start+cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            out = model(tensors["x"][batch], tensors["context"][batch], tensors["base"][batch])
            loss, parts = factorized_loss(out, tensors["y"][batch], tensors["completed"][batch],
                tensors["bid_ratio"][batch], tensors["has_bid"][batch], tensors["median_z"][batch],
                tensors["median_ratio"][batch], tensors["has_median"][batch],
                tensors["basis_z"][batch], tensors["has_basis"][batch], cfg)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss in {fit_id}, epoch {epoch}")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"Nonfinite gradient in {fit_id}, epoch {epoch}")
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optimizer.step()
            max_grad = max(max_grad, float(grad))
            total += float(loss.detach())*len(batch)
            count += len(batch)
            updates += 1
        row = {"epoch": epoch, "training_composite_loss": total/count,
               "max_gradient_norm_before_clip": max_grad, "updates": updates, "last_batch_loss_components": parts}
        if val_indices is not None:
            out = predict(model, tensors, val_indices)
            score = selection_score(out["mean"], val_indices, data, meta, cfg["horizons"])
            row["validation_MSE_RW_ratio"] = score
            if score < best-1e-10:
                best, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            traces.append(row)
            if epoch >= cfg["minimum_epochs"] and stale >= cfg["early_stopping_patience"]:
                break
        else:
            traces.append(row)
            best_epoch = epoch
    if val_indices is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter()-started
    state_path = private/(fit_id+".pt")
    torch.save({"model": model.state_dict(), "normalizer": normalizer.state(),
                "training_indices": train_indices, "config": cfg, "method": name,
                "seed": seed, "learning_rate": lr, "selected_epoch": best_epoch}, state_path)
    receipt = {"fit_id": fit_id, "method": name, "seed": seed, "learning_rate": lr,
               "training_examples": len(train_indices), "epochs_executed": len(traces),
               "updates": updates, "selected_epoch": best_epoch,
               "validation_MSE_RW_ratio": best if val_indices is not None else None,
               "elapsed_seconds": elapsed, "parameters": count_parameters(model),
               "state_sha256": sha(state_path), "traces": traces}
    (public/(fit_id+".json")).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"event": "fit_done", **{k: receipt[k] for k in
          ["fit_id", "epochs_executed", "selected_epoch", "updates", "validation_MSE_RW_ratio", "elapsed_seconds"]}}), flush=True)
    return model, tensors, receipt


def empirical_residual_atoms(mean, indices, data, meta, horizons):
    result = {}
    for horizon in horizons:
        mask = data["completed"][indices].astype(bool) & (meta.horizon.to_numpy()[indices] == horizon)
        residual = data["y"][indices[mask]]-mean[mask]
        if len(residual) < 10:
            raise ValueError("Insufficient calibration residuals")
        atoms = np.quantile(residual, (np.arange(101)+.5)/101)
        result[horizon] = atoms-atoms.mean()
    return result


def save_prediction(path, output, indices, data, meta, calibration=None):
    shared = dict(example_indices=indices, mean_z=output["mean"], y_z=data["y"][indices],
                  anchor=data["anchor"][indices], scale=data["scale"][indices])
    if "lognormal_mu_ratio" in output:
        weights = output["weights"]
        if not np.allclose(weights.sum(-1), 1, rtol=0, atol=2e-6):
            raise AssertionError("Invalid mixture probabilities")
        np.savez_compressed(path, **shared, weights=weights,
            lognormal_mu_ratio=output["lognormal_mu_ratio"], lognormal_sigma=output["lognormal_sigma"],
            mechanical_failure=output["mechanical_failure"], mean_bid_ratio=output["mean_bid_ratio"],
            bid_mixture=output["mix"], bid_concentration=output["concentration"],
            median_mean_z=output["median_mean_z"], basis_mean_z=output["basis_mean_z"])
    else:
        atoms = np.array([output["mean"][j]+calibration[int(meta.horizon.iloc[i])]
                          for j, i in enumerate(indices)])
        np.savez_compressed(path, **shared, atoms_z=atoms, weights=np.full_like(atoms,1/101))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    root = args.study_dir
    cfg = json.loads((root/"experiment_config.json").read_text())
    freeze = json.loads((root/"PROTOCOL_FREEZE.json").read_text())
    for relative, digest in freeze["frozen_sha256"].items():
        if sha(root/relative) != digest:
            raise ValueError(f"Frozen identity mismatch: {relative}")
    torch.set_num_threads(cfg["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    raw = np.load(root/"local_only/examples_development.npz")
    data = {k: raw[k] for k in raw.files}
    meta = pd.read_csv(root/"local_only/example_metadata.csv")
    if max(meta.issue_year) > 2023 or meta.target_date.max() >= "2024-01-01":
        raise ValueError("Development experiment refuses confirmation rows")
    audit = json.loads((root/"EXAMPLE_AUDIT.json").read_text())
    private = root/"local_only/neural"
    public = root/"results/neural"
    private.mkdir(exist_ok=True)
    public.mkdir(exist_ok=True)
    models = args.models or cfg["neural_models"]
    receipts, selections = [], []
    for year in cfg["evaluation_issue_years"]:
        end = f"{year}-01-01"
        tune_end = f"{year-1}-01-01"
        known_label = data["completed"].astype(bool) | data["has_bid"].astype(bool)
        initial = np.flatnonzero((meta.target_date.to_numpy() < tune_end) & known_label)
        val = np.flatnonzero((meta.issue_year.to_numpy() == year-1) & (meta.target_date.to_numpy() < end))
        final = np.flatnonzero((meta.target_date.to_numpy() < end) & known_label)
        test = np.flatnonzero(meta.issue_year.to_numpy() == year)
        assert meta.target_date.iloc[initial].max() < meta.issue_date.iloc[val].min()
        assert meta.target_date.iloc[final].max() < meta.issue_date.iloc[test].min()
        for method in models:
            columns = audit["price_only_columns"] if method == "GRU_price" else list(range(data["x"].shape[-1]))
            candidates = []
            for lr in cfg["learning_rates"]:
                fit_id = f"{method}_{year}_tune_lr{str(lr).replace('.', 'p')}_s{cfg['tuning_seed']}"
                model, tensors, receipt = fit(method, cfg["tuning_seed"], lr, cfg["max_epochs"],
                    initial, val, data, meta, cfg, columns, private, public, fit_id)
                validation_output = predict(model, tensors, val)
                np.savez_compressed(private/(fit_id+"_validation.npz"), example_indices=val,
                                    mean_z=validation_output["mean"], y_z=data["y"][val])
                candidates.append((receipt["validation_MSE_RW_ratio"], lr,
                                   receipt["selected_epoch"], validation_output))
                receipts.append(receipt)
            candidates.sort(key=lambda x: (x[0], x[1]))
            score, lr, epochs, validation_output = candidates[0]
            calibration = empirical_residual_atoms(validation_output["mean"], val, data, meta, cfg["horizons"])
            selection = {"method": method, "year": year, "learning_rate": lr,
                         "epochs": epochs, "validation_MSE_RW_ratio": score,
                         "initial_training_count": len(initial), "validation_count": len(val),
                         "final_training_count": len(final), "evaluation_count": len(test)}
            selections.append(selection)
            (public/f"selection_{method}_{year}.json").write_text(json.dumps(selection, indent=2))
            for seed in cfg["refit_seeds"]:
                fit_id = f"{method}_{year}_refit_s{seed}"
                model, tensors, receipt = fit(method, seed, lr, epochs, final, None, data,
                                              meta, cfg, columns, private, public, fit_id)
                output = predict(model, tensors, test)
                save_prediction(private/f"{method}_{year}_s{seed}.npz", output, test, data, meta, calibration)
                receipts.append(receipt)
            # Completion checkpoint is aggregate metadata only; all scored files
            # remain private and are independently analyzed after the full run.
            (public/"RUN_PROGRESS.json").write_text(json.dumps({"completed_selections": selections,
                "completed_fits": len(receipts), "updates": sum(r["updates"] for r in receipts)}, indent=2))
    summary = {"status": "COMPLETE", "models": models, "fits": len(receipts),
               "refits": sum("refit" in r["fit_id"] for r in receipts),
               "updates": sum(r["updates"] for r in receipts),
               "fit_seconds": sum(r["elapsed_seconds"] for r in receipts),
               "selections": selections, "protocol_freeze_sha256": sha(root/"PROTOCOL_FREEZE.json"),
               "failures": 0, "torch": torch.__version__}
    (public/"RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
