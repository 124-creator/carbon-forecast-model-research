"""Predeclared raw-GRU and published SWEWT-GRU controlled references.

Both use the exact stage-e canonical preparation and only three supervised
heads. This is not an author-code/numerical reproduction of Energies 2024.
Raw row-level sources, predictions and binary checkpoints remain in --out.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import inspect
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ewtpy import EWT1D
from torch import nn

import run_signed_response_search as core

SOURCE = Path(__file__).resolve().parent
FAMILIES = ("raw_gru", "swewt_gru")
SEEDS = (17, 29, 43)
LRS = (.003, .01)
MAX_EPOCHS, MIN_EPOCHS, PATIENCE, BATCH = 80, 15, 12, 128
PAPER_DOI = "10.3390/en17174358"


def price_features(price, origin, scale):
    history = np.asarray(price[origin-63:origin+1], dtype=float).copy()
    assert history.shape == (64,) and np.isfinite(history).all()
    bands, _, _ = EWT1D(history, N=3, log=0, detect="locmax", completion=1,
                        reg="average", lengthFilter=10, sigmaFilter=5)
    assert bands.shape == (64, 3) and np.isfinite(bands).all()
    bands = bands[-17:].copy()
    bands[:, 0] -= price[origin]
    return (np.column_stack((history[-17:]-price[origin], bands))/scale).astype("float32")


def sequences(d, family):
    if family == "raw_gru":
        return d["x"].astype("float32")
    if family != "swewt_gru":
        raise ValueError(family)
    channels = np.stack([price_features(d["price"], int(t), d["scale"]) for t in d["origins"]])
    return np.concatenate((channels, d["x"][:, -17:, 1:].astype("float32")), axis=-1)


class ReferenceGRU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.GRU(channels, 32, batch_first=True)
        self.head = nn.Linear(32, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        z, _ = self.encoder(x)
        return self.head(z[:, -1])


def make_model(family, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return ReferenceGRU(3 if family == "raw_gru" else 6)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+256])).numpy()
                               for i in range(0, len(x), 256)])


def fit(d, x, family, seed, lr, subset, *, epochs=MAX_EPOCHS, selection=True):
    model = make_model(family, seed)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    ix = np.flatnonzero(d["masks"][subset])
    val = d["masks"]["val"]
    y = (d["y"]/d["scale"]).astype("float32")
    variance = (d["variance"]/d["scale"]**2).astype("float32")
    weights = torch.from_numpy(variance)
    best_epoch, stale, positive_steps = 0, 0, 0
    init_val = float(np.mean((predict(model, x[val])-y[val])**2/variance)) if selection else None
    best = init_val if selection else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        total, count = 0., 0
        for batch in np.array_split(rng.permutation(ix), max(1, int(np.ceil(len(ix)/BATCH)))):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(torch.from_numpy(x[batch]))-torch.from_numpy(y[batch]))**2/weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite neural-reference loss")
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
            if not np.isfinite(grad):
                raise FloatingPointError("Nonfinite neural-reference gradient")
            positive_steps += int(grad > 0)
            opt.step()
            total += float(loss.detach())*len(batch)
            count += len(batch)
        score = float(np.mean((predict(model, x[val])-y[val])**2/variance)) if selection else None
        history.append({"epoch": epoch, "train_loss": total/count, "validation_loss": score})
        if selection:
            if score < best-1e-7:
                best, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                break
        else:
            best_epoch = epoch
    attempt_change = float(torch.sqrt(sum(torch.sum((v.detach()-initial[k]).double()**2)
                                         for k, v in model.state_dict().items())))
    if selection:
        model.load_state_dict(best_state)
    receipt = {"family": family, "seed": seed, "learning_rate": lr, "subset": subset,
               "selection": selection, "initial_validation_loss": init_val,
               "validation_loss": best if selection else None,
               "selected_epoch": best_epoch, "epochs_executed": len(history),
               "positive_gradient_steps": positive_steps,
               "attempted_parameter_change_l2": attempt_change,
               "fitted_parameter_count": sum(p.numel() for p in model.parameters()),
               "heads": 3, "all_heads_supervised": True, "history": history,
               "state_dict": {k: v.detach().tolist() for k, v in model.state_dict().items()}}
    return model, receipt


def run_job(spec, family, out):
    torch.set_num_threads(1)
    d = core.prepare(spec)
    x = sequences(d, family)
    dest = Path(out)/f"{spec['market']}_{core.FOLD}"/family
    dest.mkdir(parents=True, exist_ok=False)
    choices = []
    for lr in LRS:
        model, receipt = fit(d, x, family, 17, lr, "train")
        name = f"selection_lr{lr:g}"
        core.write_json(dest/(name+".json"), receipt)
        torch.save(model.state_dict(), dest/(name+".pt"))
        choices.append({"learning_rate": lr, "validation_loss": receipt["validation_loss"],
                        "selected_epoch": receipt["selected_epoch"], "receipt": name+".json"})
    selected = min(choices, key=lambda r: (r["validation_loss"], r["learning_rate"], r["selected_epoch"]))
    core.write_json(dest/"SELECTION.json", {"selected": selected, "trials": choices})
    pred, receipts = [], []
    for seed in SEEDS:
        model, receipt = fit(d, x, family, seed, selected["learning_rate"], "final",
                             epochs=selected["selected_epoch"], selection=False)
        core.write_json(dest/f"refit_seed{seed}.json", receipt)
        torch.save(model.state_dict(), dest/f"refit_seed{seed}.pt")
        pred.append(predict(model, x[d["masks"]["test"]])*d["scale"])
        receipts.append(receipt)
    pred = np.asarray(pred, dtype=float)
    observed = d["y"][d["masks"]["test"]]
    np.savez_compressed(dest/"predictions.npz", predictions=pred, observed=observed,
                        origins=d["origins"][d["masks"]["test"]], horizons=np.array(core.HORIZONS),
                        seeds=np.array(SEEDS), deterministic=np.array([False]))
    rw = np.mean(observed**2, axis=0)
    mse = np.mean((pred-observed[None])**2, axis=(0, 1))
    summary = {"market": spec["market"], "role": spec["role"], "fold": core.FOLD,
               "family": family, "n_origins": len(observed), "horizons": list(core.HORIZONS),
               "mse_by_horizon": mse.tolist(), "rw_mse_by_horizon": rw.tolist(),
               "mse_over_rw_by_horizon": (mse/rw).tolist(),
               "mean_mse_over_rw": float(np.mean(mse/rw)), "seed_count": len(SEEDS),
               "primary_seed_summary": "average squared error across seeds before normalizing by RW",
               "ensemble_mean_mse_over_rw": float(np.mean(np.mean((pred.mean(0)-observed)**2, axis=0)/rw)),
               "selected_epoch": selected["selected_epoch"],
               "fitted_parameter_count": receipts[0]["fitted_parameter_count"],
               "refit_positive_gradient_steps": sum(r["positive_gradient_steps"] for r in receipts)}
    core.write_json(dest/"SUMMARY.json", summary)
    return summary


def dependencies():
    ewt_source = Path(inspect.getsourcefile(EWT1D))
    return {"ewtpy_version": importlib.metadata.version("ewtpy"),
            "ewtpy_source_sha256": core.digest(ewt_source), "ewtpy_source": str(ewt_source),
            "numpy": np.__version__, "torch": torch.__version__}


def preflight(specs):
    torch.set_num_threads(1)
    checks = []
    for spec in specs:
        d = core.prepare(spec)
        t = int(d["origins"][d["masks"]["val"]][0])
        future_changed = d["price"].copy()
        future_changed[t+1:] += 99999.
        gap = float(np.max(np.abs(price_features(d["price"], t, d["scale"])
                                  -price_features(future_changed, t, d["scale"]))))
        assert gap == 0.
        shapes = {}
        for family in FAMILIES:
            xb = d["x"][:2].astype("float32") if family == "raw_gru" else np.concatenate((
                np.stack([price_features(d["price"], int(o), d["scale"]) for o in d["origins"][:2]]),
                d["x"][:2, -17:, 1:].astype("float32")), axis=-1)
            model = make_model(family, 17)
            p = predict(model, xb)
            assert p.shape == (2, 3) and np.array_equal(p, np.zeros_like(p))
            shapes[family] = {"shape": list(xb.shape), "parameters": sum(v.numel() for v in model.parameters())}
        checks.append({"market": spec["market"], "future_ewt_perturbation_max_gap": gap, "forward": shapes})
    return {"status": "PASS_NO_MARKET_FIT_NO_SCORE", "market_fits": 0, "performance_scores": 0,
            "checks": checks, "dependencies": dependencies()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", required=True)
    p.add_argument("--out")
    p.add_argument("--plan")
    p.add_argument("--expected-plan-sha256")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--preflight-report")
    args = p.parse_args()
    specs = core.read_datasets(args.datasets)
    if args.preflight_report:
        if Path(args.preflight_report).exists():
            raise FileExistsError("Use a new preflight path")
        core.write_json(args.preflight_report, preflight(specs))
        print("PASS_NO_MARKET_FIT_NO_SCORE", flush=True)
        return
    if not args.out or not args.plan or not args.expected_plan_sha256:
        p.error("A run requires --out, --plan and --expected-plan-sha256")
    assert core.digest(args.plan) == args.expected_plan_sha256
    core.check_old_sources()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("New empty output directory required")
    out.mkdir(parents=True, exist_ok=True)
    before = {**core.source_hashes(), Path(__file__).name: core.digest(__file__)}
    protocol = {"created_unix": time.time(), "stage": "e_prespecified_neural_references",
                "datasets": specs, "source_sha256": before,
                "statistical_plan_sha256": args.expected_plan_sha256,
                "paper_doi": PAPER_DOI, "classification": "controlled multistep adaptation, not author numerical reproduction",
                "families": list(FAMILIES), "learning_rates": LRS, "selection_seed": 17,
                "refit_seeds": SEEDS, "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS,
                "patience": PATIENCE, "batch_size": BATCH, "horizons": list(core.HORIZONS),
                "optimizer": "AdamW, weight_decay1e-4, gradient_clip5", "early_stop_tolerance": 1e-7,
                "input": {"raw_gru": "64 price-change/volume-surprise/valid tokens",
                          "swewt_gru": "64 observed price levels separately EWT3; last17 raw-origin, low-origin, band2, band3 plus volume-surprise/valid"},
                "ewt": {"N": 3, "log": 0, "detect": "locmax", "completion": 1, "reg": "average", "lengthFilter": 10, "sigmaFilter": 5},
                "heads": "Three zero-initialized heads, all supervised H1/5/20; same normalized target loss as core",
                "departures_from_paper": ["Different markets and 60/20/20 chronological split", "Three horizons and volume context", "Fixed window64 and mode3, GRU32; two learning rates and max80 instead of published TPE10 and max1000", "Origin centering and train-prefix RMS rather than paper minmax"],
                "planned_selection_fits": 2*len(FAMILIES)*len(specs),
                "planned_final_fits": 3*len(FAMILIES)*len(specs), "dependencies": dependencies()}
    core.write_json(out/"PROTOCOL_BEFORE_RUN.json", protocol)
    for spec in specs:
        core.write_json(out/f"{spec['market']}_{core.FOLD}"/"INPUT_MANIFEST.json", core.prepare(spec)["manifest"])
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_job, spec, family, str(out)) for spec in specs for family in FAMILIES]
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            print(f"COMPLETE {result['market']} {result['family']} Q={result['mean_mse_over_rw']:.9f}", flush=True)
    assert before == {**core.source_hashes(), Path(__file__).name: core.digest(__file__)}
    assert core.digest(args.plan) == args.expected_plan_sha256
    core.check_old_sources()
    core.write_json(out/"SUMMARY.json", sorted(results, key=lambda r: (r["market"], r["family"])))
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in results]).to_csv(out/"SUMMARY.csv", index=False)
    core.write_json(out/"EXECUTION_COMPLETE.json", {"status": "completed", "summary_rows": len(results),
                    "planned_fit_count": len(specs)*len(FAMILIES)*(len(LRS)+len(SEEDS)),
                    "source_hashes_unchanged": True, "plan_hash_unchanged": True})


if __name__ == "__main__":
    main()
