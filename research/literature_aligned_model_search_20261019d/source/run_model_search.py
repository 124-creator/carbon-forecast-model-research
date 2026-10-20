"""One common development backtest for the shock-allocation model and controls.

All 2022/2023 results are development evidence: these years have been inspected
in earlier project studies.  No historical-vintage or fresh holdout claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn


MARKETS = ["GUANGZHOU", "HUBEI", "SHANGHAI_D4", "EU_REPORTED"]
YEARS = [2022, 2023]
SEEDS = [17, 29, 43]
HORIZONS = np.array([1, 5, 20])
LOOKBACK = 64
ENERGY_LAG_DAYS = 14
FAMILIES = ["allocation_full", "allocation_constant", "allocation_readout",
            "allocation_price", "gru", "lstm"]
LRS = [0.003, 0.01]
MAX_EPOCHS, MIN_EPOCHS, PATIENCE, BATCH_SIZE = 80, 15, 12, 128
STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_arrays(market):
    path = REPO / "research/cross_market_innovation_20261017/data" / f"{market}_development.csv"
    df = pd.read_csv(path).sort_values("date").reset_index(drop=True)
    assert df.date.is_unique and np.isfinite(df.price).all()
    dates = pd.to_datetime(df.date).to_numpy(dtype="datetime64[D]")
    price = df.price.to_numpy(float)
    tokens = [np.r_[0.0, np.diff(price)]]
    sources, alignments = [{"path": str(path.relative_to(REPO)), "sha256": digest(path)}], []
    for name in ["brent", "wti", "henryhub"]:
        source = REPO / "data/parsed" / f"{name}_eia_daily.csv"
        ef = pd.read_csv(source).sort_values("date")
        missing_count = int(ef.spot_price.isna().sum())
        ef = ef.dropna(subset=["spot_price"])
        ed = pd.to_datetime(ef.date).to_numpy(dtype="datetime64[D]")
        ep = ef.spot_price.to_numpy(float)
        j = np.searchsorted(ed, dates-np.timedelta64(ENERGY_LAG_DAYS, "D"), side="right")-1
        assert (j >= 0).all() and np.isfinite(ep[j]).all()
        assert (ed[j] <= dates-np.timedelta64(ENERGY_LAG_DAYS, "D")).all()
        # Arithmetic changes preserve the negative WTI price in April 2020.
        tokens.append(np.r_[0.0, np.diff(ep[j])])
        sources.append({"path": str(source.relative_to(REPO)), "sha256": digest(source)})
        alignments.append({"series": name, "minimum_age_days": int(np.min((dates-ed[j]).astype(int))),
                           "maximum_age_days": int(np.max((dates-ed[j]).astype(int))),
                           "source_missing_price_rows_excluded_before_backward_asof": missing_count})
    return {"dates": dates, "price": price, "tokens": np.column_stack(tokens),
            "sources": sources, "alignment": alignments}


def prepare(market, year):
    d = load_arrays(market)
    dates, p, tokens = d["dates"], d["price"], d["tokens"]
    years = pd.DatetimeIndex(dates).year.to_numpy()
    first = int(np.flatnonzero(years == year)[0])
    val_start, val_stop = first-20-80, first-20
    origins = np.arange(LOOKBACK, len(p)-20)
    train = (origins+20 < val_start)
    val = (origins >= val_start) & (origins < val_stop)
    final = origins+20 < first
    test = (years[origins] == year) & (years[origins+20] == year)
    assert train.sum() >= 200 and val.sum() == 80 and test.sum() >= 50
    assert origins[train][-1]+20 < origins[val][0]
    assert origins[val][-1]+20 < origins[test][0]
    scale = np.sqrt(np.mean(tokens[1:val_start]**2, axis=0)).clip(1e-7)
    x = np.stack([tokens[t-LOOKBACK+1:t+1]/scale for t in origins]).astype("float32")
    y = np.stack([p[origins+h]-p[origins] for h in HORIZONS], axis=1)
    variance = np.mean(y[train]**2, axis=0).clip(scale[0]**2*1e-6)
    return {**d, "origins": origins, "x": x, "y": y,
            "masks": {"train": train, "val": val, "final": final, "test": test},
            "scale": scale, "variance": variance,
            "manifest": {"market": market, "year": year,
                         "counts": {k: int(v.sum()) for k,v in {"train":train,"val":val,"final":final,"test":test}.items()},
                         "last_train_label": str(dates[origins[train][-1]+20]),
                         "first_validation_origin": str(dates[origins[val][0]]),
                         "last_validation_label": str(dates[origins[val][-1]+20]),
                         "first_test_origin": str(dates[origins[test][0]]),
                         "last_test_label": str(dates[origins[test][-1]+20]),
                         "training_rms": scale.tolist(), "train_target_second_moment": variance.tolist(),
                         "sources": d["sources"], "energy_alignment": d["alignment"]}}


class DirectRecurrent(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.encoder = (nn.GRU if kind == "gru" else nn.LSTM)(4, 32, batch_first=True)
        self.readout = nn.Linear(32, 20)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x):
        z, _ = self.encoder(x)
        return self.readout(z[:, -1])


def make_model(family, seed):
    from innovation_models import ShockAllocationForecaster
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if family in ["gru", "lstm"]:
        return DirectRecurrent(family)
    variant = {"allocation_full": "full", "allocation_constant": "constant_allocation",
               "allocation_readout": "readout_gate", "allocation_price": "full"}[family]
    return ShockAllocationForecaster(n_exog=0 if family == "allocation_price" else 3,
                                    horizon=20, hidden=16, modes=2, variant=variant)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[j:j+256])).numpy()[:, HORIZONS-1]
                               for j in range(0, len(x), 256)])


def train_fit(family, seed, lr, d, which, epochs, early_stop, trace_path, checkpoint_path):
    started = time.monotonic()
    model = make_model(family, seed)
    x = d["x"].copy()
    if family == "allocation_price":
        x = x[:, :, :1].copy()
    ym = d["y"]/d["scale"][0]
    vm = d["variance"]/d["scale"][0]**2
    train = np.flatnonzero(d["masks"][which])
    val = d["masks"]["val"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    weights = torch.from_numpy(vm.astype("float32"))
    history, best_epoch, stale = [], 0, 0
    initial_val = float(np.mean((predict(model, x[val])-ym[val])**2/vm)) if early_stop else None
    best_score = initial_val if early_stop else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    initial_state = {k: v.detach().clone() for k,v in model.state_dict().items()}
    gradient_max, gradient_positive_steps = 0.0, 0
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for ix in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train)/BATCH_SIZE)))):
            xb, yb = torch.from_numpy(x[ix]), torch.from_numpy(ym[ix].astype("float32"))
            opt.zero_grad(set_to_none=True)
            pred = model(xb)[:, HORIZONS-1]
            loss = torch.mean((pred-yb)**2/weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss {family}, {seed}, {lr}")
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            assert np.isfinite(grad)
            gradient_max = max(gradient_max, grad)
            gradient_positive_steps += int(grad > 0)
            opt.step()
            losses.append(float(loss.detach()))
        val_score = float(np.mean((predict(model, x[val])-ym[val])**2/vm)) if early_stop else None
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_score})
        if early_stop:
            if val_score < best_score-1e-7:
                best_score, best_epoch, stale = val_score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                break
        else:
            best_epoch = epoch
    change = float(sum((v.detach()-initial_state[k]).double().pow(2).sum() for k,v in model.state_dict().items()
                       if v.is_floating_point())**0.5)
    if early_stop:
        model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    receipt = {"family": family, "seed": seed, "learning_rate": lr, "mode": which,
               "parameters": sum(p.numel() for p in model.parameters()),
               "epochs_executed": len(history), "selected_epoch": best_epoch,
               "initial_validation_loss": initial_val,
               "selected_validation_loss": best_score if early_stop else None,
               "gradient_positive_steps": gradient_positive_steps, "max_gradient_norm_before_clip": gradient_max,
               "optimization_parameter_change_l2_before_selection": change,
               "elapsed_seconds": time.monotonic()-started, "history": history,
               "checkpoint_sha256": digest(checkpoint_path)}
    write_json(trace_path, receipt)
    return model, receipt


def neural_job(market, year, family, out):
    torch.set_num_threads(1)
    d = prepare(market, year)
    dest = Path(out) / f"{market}_{year}" / family
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for lr in LRS:
        _, rec = train_fit(family, 17, lr, d, "train", MAX_EPOCHS, True,
                           dest/f"selection_lr{lr}.json", dest/f"selection_lr{lr}.pt")
        records.append(rec)
    choice = min(records, key=lambda v: (v["selected_validation_loss"], v["learning_rate"]))
    write_json(dest/"CHOICE_BEFORE_TEST.json", {k:v for k,v in choice.items() if k != "history"})
    preds = []
    for seed in SEEDS:
        model, rec = train_fit(family, seed, choice["learning_rate"], d, "final", choice["selected_epoch"], False,
                               dest/f"refit_seed{seed}.json", dest/f"refit_seed{seed}.pt")
        xt = d["x"][d["masks"]["test"]].copy()
        if family == "allocation_price":
            xt = xt[:, :, :1].copy()
        pred = predict(model, xt)*d["scale"][0]
        preds.append(pred)
    mask = d["masks"]["test"]
    np.savez_compressed(dest/"predictions.npz", predictions=np.stack(preds),
                        observed=d["y"][mask], origins=d["origins"][mask],
                        horizons=HORIZONS, seeds=np.array(SEEDS))
    rec = summarize_predictions(market, year, family, np.stack(preds), d["y"][mask])
    rec.update({"selected_lr": choice["learning_rate"], "selected_epoch": choice["selected_epoch"],
                "parameters": choice["parameters"]})
    write_json(dest/"SUMMARY.json", rec)
    return rec


def summarize_predictions(market, year, family, predictions, observed):
    rw = np.mean(observed**2, axis=0)
    seed_scores = np.mean((predictions-observed[None])**2, axis=1)/rw[None]
    ensemble = predictions.mean(axis=0)
    return {"market": market, "year": year, "family": family,
            "n_origins": len(observed), "rw_mse_by_horizon": rw.tolist(),
            "seed_mse_over_rw_by_horizon": seed_scores.tolist(),
            "mean_seed_mse_over_rw": float(seed_scores.mean()),
            "ensemble_mse_over_rw_by_horizon": (np.mean((ensemble-observed)**2, axis=0)/rw).tolist(),
            "ensemble_mse_over_rw": float(np.mean(np.mean((ensemble-observed)**2, axis=0)/rw))}


def classical_job(market, year, out):
    d = prepare(market, year)
    m = d["masks"]
    y, variance = d["y"], d["variance"]
    dest = Path(out)/f"{market}_{year}"
    results = []
    for family in ["rw", "drift", "ridge_price", "ridge_full", "hgb_full"]:
        fd = dest/family
        fd.mkdir(parents=True, exist_ok=True)
        if family == "rw":
            pred, choice = np.zeros_like(y[m["test"]]), {
                "configuration": "no change", "validation_loss": float(np.mean(y[m["val"]]**2/variance))}
        elif family == "drift":
            candidates = []
            r = d["tokens"][:, 0]
            for window in [20, 60, 120, None]:
                pp = np.array([r[max(1,t-window+1) if window else 1:t+1].mean()*HORIZONS for t in d["origins"]])
                score = float(np.mean((pp[m["val"]]-y[m["val"]])**2/variance))
                candidates.append((score, str(window), pp))
            best = min(candidates, key=lambda z: (z[0], z[1]))
            pred, choice = best[2][m["test"]], {"window": best[1], "validation_loss": best[0]}
        elif family.startswith("ridge"):
            xx = d["x"][:, -20:, :1] if family == "ridge_price" else d["x"]
            xx = xx.reshape(len(xx), -1)
            scaler = StandardScaler().fit(xx[m["train"]])
            xx = scaler.transform(xx)
            trials = []
            for alpha in [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]:
                model = Ridge(alpha=alpha).fit(xx[m["train"]], y[m["train"]])
                score = float(np.mean((model.predict(xx[m["val"]])-y[m["val"]])**2/variance))
                trials.append((score, alpha))
            score, alpha = min(trials)
            model = Ridge(alpha=alpha).fit(xx[m["final"]], y[m["final"]])
            pred, choice = model.predict(xx[m["test"]]), {"alpha": alpha, "validation_loss": score, "trials": trials}
        else:
            xx = d["x"][:, -20:].reshape(len(y), -1)
            trials = []
            for leaves in [3, 7]:
                for l2 in [1.0, 10.0]:
                    models = [HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=leaves, min_samples_leaf=25,
                              l2_regularization=l2, learning_rate=0.05, early_stopping=False, random_state=17)
                              .fit(xx[m["train"]], y[m["train"], h]) for h in range(3)]
                    vp = np.column_stack([model.predict(xx[m["val"]]) for model in models])
                    trials.append((float(np.mean((vp-y[m["val"]])**2/variance)), leaves, l2))
            score, leaves, l2 = min(trials)
            models = [HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=leaves, min_samples_leaf=25,
                      l2_regularization=l2, learning_rate=0.05, early_stopping=False, random_state=17)
                      .fit(xx[m["final"]], y[m["final"], h]) for h in range(3)]
            pred = np.column_stack([model.predict(xx[m["test"]]) for model in models])
            choice = {"max_leaf_nodes": leaves, "l2_regularization": l2, "validation_loss": score, "trials": trials}
        write_json(fd/"SELECTION.json", choice)
        np.savez_compressed(fd/"predictions.npz", predictions=pred[None], observed=y[m["test"]],
                            origins=d["origins"][m["test"]], horizons=HORIZONS, seeds=np.array([-1]))
        rec = summarize_predictions(market, year, family, pred[None], y[m["test"]])
        write_json(fd/"SUMMARY.json", rec)
        results.append(rec)
    return results


def freeze(out, markets, years):
    out = Path(out)
    if (out/"PROTOCOL_BEFORE_RUN.json").exists():
        raise RuntimeError("Output exists; preserve previous experiment. Use a new output directory.")
    sources = {name: digest(Path(__file__).parent/name) for name in ["run_model_search.py", "innovation_models.py"]}
    protocol = {"stage": "development_search", "exposure": "2022/2023 already used in prior studies",
                "markets": markets, "years": years, "horizons_observation_steps": HORIZONS.tolist(),
                "lookback": LOOKBACK, "validation_origins": 80, "label_purge": 20,
                "neural_families": FAMILIES, "learning_rates": LRS,
                "selection_seed": 17, "refit_seeds": SEEDS, "maximum_epochs": MAX_EPOCHS,
                "minimum_epochs_before_stopping": MIN_EPOCHS, "patience": PATIENCE,
                "optimizer": "AdamW(weight_decay=1e-4), global gradient norm clip 5",
                "batch_size": BATCH_SIZE,
                "training_loss": "mean native price squared error / training target second moment, equal H1/H5/H20",
                "selection": "past validation only; select lr and epoch on seed17, refit all matured pretest labels for same epochs with seeds17/29/43",
                "test_updates": "weights frozen during each year; each origin receives actual past observations",
                "primary_comparison": "equal market/year/horizon mean MSE ratio to RW; same-data structural controls",
                "primary_seed_summary": "mean_seed_mse_over_rw: score each refit seed then average losses; do not score a seed ensemble for the primary claim",
                "secondary_seed_summary": "ensemble_mse_over_rw: average the three seed forecasts before scoring, explicitly a three-model ensemble",
                "strong_baseline_selector": "per market/year choose lowest past validation loss from RW, drift, ridge_price, ridge_full, hgb_full, GRU, LSTM; evaluate that family's final forecast without consulting test performance",
                "uncertainty": "paired block bootstrap; seeds measure algorithm randomness, not independent markets",
                "energy_features": {"lag_calendar_days": ENERGY_LAG_DAYS,
                                    "version": "public historical file; not historical-vintage certified",
                                    "transform": "arithmetic change between aligned carbon observation dates"},
                "gate": "no universal publication percentage; need consistent gain over RW and strongest same-data control, material allocation ablation and supported novelty before confirmation",
                "source_sha256": sources, "created_unix": time.time()}
    write_json(out/"PROTOCOL_BEFORE_RUN.json", protocol)
    for market in markets:
        for year in years:
            d = prepare(market, year)
            write_json(out/f"{market}_{year}"/"INPUT_MANIFEST.json", d["manifest"])
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markets", nargs="+", default=MARKETS)
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    protocol = freeze(args.out, args.markets, args.years)
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for market in args.markets:
            for year in args.years:
                futures[pool.submit(classical_job, market, year, str(args.out))] = f"{market}/{year}/classical"
                for family in FAMILIES:
                    futures[pool.submit(neural_job, market, year, family, str(args.out))] = f"{market}/{year}/{family}"
        for fut in as_completed(futures):
            result = fut.result()
            rr = result if isinstance(result, list) else [result]
            summaries.extend(rr)
            print(json.dumps({"completed": futures[fut], "scores": [{k:r[k] for k in ["family","mean_seed_mse_over_rw","ensemble_mse_over_rw"]} for r in rr]}), flush=True)
            write_json(args.out/"PROGRESS_SUMMARIES.json", summaries)
    write_json(args.out/"ALL_SUMMARIES.json", summaries)
    write_json(args.out/"EXECUTION_COMPLETE.json", {"status":"completed", "jobs":len(futures), "summary_rows":len(summaries),
                                                    "source_hashes_unchanged": protocol["source_sha256"] == {name:digest(Path(__file__).parent/name) for name in protocol["source_sha256"]}})


if __name__ == "__main__":
    main()
