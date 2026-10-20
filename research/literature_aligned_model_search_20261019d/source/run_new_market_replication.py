"""Fixed-candidate replication on three previously unmodeled ScienceDB markets.

The target is the authors' daily transaction mean, not a closing price.  The
original 2026-10-17 60/20/20 row boundaries are preserved.  Prior mechanical
data inspection is disclosed; these are not never-accessed source values.
This adapter leaves both earlier model implementations and runners unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_model_search as common
import run_liquidity_extension as liquidity
from innovation_models import ShockAllocationForecaster


DATA_ROOT = Path(os.environ.get("CARBON_SCIENCEDB_DATA_DIR", str(common.STUDY / "data/sciencedb_819_v1/prepared")))
MARKETS = ["BEIJING_MEAN", "TIANJIN_MEAN", "CHONGQING_MEAN"]
FOLD = "fixed602020"
FAMILIES = ["liquidity_full", "liquidity_readout", "shock_size_only",
            "liquidity_unconstrained", "allocation_constant", "allocation_readout", "gru", "lstm"]
PINNED = {
    "BEIJING_MEAN": (673, 403, 537, "952a69168d396cce3bbd54a5704f861cc459f67e9e121434fd7672e86bf18ed3"),
    "TIANJIN_MEAN": (457, 274, 365, "c35e24f047d903f1d1f68513e87900891a89314b0fb91100d57390c34bc27dda"),
    "CHONGQING_MEAN": (240, 144, 192, "0782976147c5e6948a494f34990331e58b2815dcef946fb1fad1ff3fef632e67"),
}


def load_replication(market):
    path = DATA_ROOT / f"{market}.csv"
    n, _, _, expected = PINNED[market]
    assert common.digest(path) == expected, "Published input identity differs from the preserved source"
    df = pd.read_csv(path)
    assert len(df) == n and df.date.is_unique and df.date.is_monotonic_increasing
    dates = pd.to_datetime(df.date).to_numpy(dtype="datetime64[D]")
    price = df.price.to_numpy(float)
    volume = df.volume_tons.to_numpy(float)
    assert np.isfinite(price).all() and (price > 0).all()
    assert np.isfinite(volume).all() and (volume > 0).all()
    tokens = [np.r_[0.0, np.diff(price)]]
    sources = [{"path": str(path), "sha256": expected, "doi": "10.11922/sciencedb.819"}]
    alignments = []
    for name in ["brent", "wti", "henryhub"]:
        source = common.REPO / "data/parsed" / f"{name}_eia_daily.csv"
        ef = pd.read_csv(source).sort_values("date")
        missing = int(ef.spot_price.isna().sum())
        ef = ef.dropna(subset=["spot_price"])
        ed = pd.to_datetime(ef.date).to_numpy(dtype="datetime64[D]")
        ep = ef.spot_price.to_numpy(float)
        j = np.searchsorted(ed, dates - np.timedelta64(common.ENERGY_LAG_DAYS, "D"), side="right") - 1
        assert (j >= 0).all() and np.isfinite(ep[j]).all()
        assert (ed[j] <= dates - np.timedelta64(common.ENERGY_LAG_DAYS, "D")).all()
        tokens.append(np.r_[0.0, np.diff(ep[j])])
        sources.append({"path": str(source.relative_to(common.REPO)), "sha256": common.digest(source)})
        alignments.append({"series": name, "minimum_age_days": int(np.min((dates-ed[j]).astype(int))),
                           "maximum_age_days": int(np.max((dates-ed[j]).astype(int))),
                           "source_missing_price_rows_excluded_before_backward_asof": missing})
    logs = np.log1p(volume)
    context = np.zeros((len(df), 2), dtype="float32")
    for t in range(10, len(df)):
        past = logs[max(0, t-60):t]
        context[t, 0] = np.clip((logs[t]-np.median(past))/max(float(np.std(past)), .25), -5, 5)
        context[t, 1] = 1.0
    return {"dates": dates, "price": price, "tokens": np.column_stack(tokens),
            "volume_context": context, "sources": sources, "alignment": alignments}


def prepare_replication(market, fold=FOLD):
    assert fold == FOLD, "This experiment only accepts its preserved fixed split"
    d = load_replication(market)
    n, train_end, validation_end, _ = PINNED[market]
    dates, p, tokens = d["dates"], d["price"], d["tokens"]
    origins = np.arange(common.LOOKBACK, n-20)
    masks = {
        "train": origins+20 < train_end,
        "val": (origins >= train_end) & (origins+20 < validation_end),
        "final": origins+20 < validation_end,
        "test": origins >= validation_end,
    }
    assert origins[masks["train"]][-1]+20 < origins[masks["val"]][0]
    assert origins[masks["val"]][-1]+20 < origins[masks["test"]][0]
    scale = np.sqrt(np.mean(tokens[1:train_end]**2, axis=0)).clip(1e-7)
    x = np.stack([tokens[t-common.LOOKBACK+1:t+1]/scale for t in origins]).astype("float32")
    volume_windows = np.stack([d["volume_context"][t-common.LOOKBACK+1:t+1] for t in origins])
    x = np.concatenate([x, volume_windows], axis=-1)
    y = np.stack([p[origins+h]-p[origins] for h in common.HORIZONS], axis=1)
    variance = np.mean(y[masks["train"]]**2, axis=0).clip(scale[0]**2*1e-6)
    manifest = {
        "market": market, "fold": fold, "legacy_year_field_is_fold_id": True,
        "target": "author-curated daily transaction mean in CNY/tonne",
        "row_boundaries": [0, train_end, validation_end, n],
        "counts": {k: int(v.sum()) for k, v in masks.items()},
        "origin_ranges": {k: [int(origins[v][0]), int(origins[v][-1])] for k,v in masks.items()},
        "first_train_origin": str(dates[origins[masks["train"]][0]]),
        "last_train_label": str(dates[origins[masks["train"]][-1]+20]),
        "first_validation_origin": str(dates[origins[masks["val"]][0]]),
        "last_validation_label": str(dates[origins[masks["val"]][-1]+20]),
        "first_test_origin": str(dates[origins[masks["test"]][0]]),
        "last_test_label": str(dates[origins[masks["test"]][-1]+20]),
        "training_rms": scale.tolist(), "train_target_second_moment": variance.tolist(),
        "sources": d["sources"], "energy_alignment": d["alignment"],
        "volume": {"unit": "tonnes", "valid_rows": n, "usable_context_rows": n-10,
                   "normalization": "current log1p volume minus prior-60-record median, divided by prior std floored .25, clipped [-5,5]; >=10 prior records",
                   "information": "current record's price and volume assumed observed at forecast issue; no turnover input",
                   "source_processing": "author monthly-reference outlier screening/replacement; flags and point-in-time vintage unavailable"},
        "role": "small-sample stress test" if market == "CHONGQING_MEAN" else "prespecified primary cross-market replication",
    }
    return {**d, "origins": origins, "x": x, "y": y, "masks": masks,
            "scale": scale, "variance": variance, "manifest": manifest}


class PriceEnergyView(nn.Module):
    """A parameter-free input view for previously fixed price/energy controls."""
    def __init__(self, variant):
        super().__init__()
        self.core = ShockAllocationForecaster(n_exog=3, horizon=20, hidden=16, modes=2, variant=variant)

    def forward(self, x):
        return self.core(x[:, :, :4])


def make_replication_model(family, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if family in ["allocation_constant", "allocation_readout"]:
        return PriceEnergyView("constant_allocation" if family == "allocation_constant" else "readout_gate")
    return liquidity.make_liquidity_model(family, seed)


def freeze_replication(out, markets):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Preserve prior output; replication requires a new empty destination")
    source_names = ["run_model_search.py", "innovation_models.py", "run_liquidity_extension.py",
                    "liquidity_models.py", Path(__file__).name]
    stat_path = common.STUDY / "audits/V3_STATISTICAL_AUDIT_PLAN.json"
    assert stat_path.exists(), "Freeze the independent analysis plan before starting market fits"
    protocol = {
        "stage": "fixed_candidate_new_market_replication_after_v1_v2_development",
        "created_unix": time.time(), "markets": markets, "fold": FOLD,
        "source_doi": "10.11922/sciencedb.819", "license": "CC BY 4.0",
        "target": "author-curated daily transaction mean price, CNY/tonne; not close",
        "prior_exposure": "source values previously mechanically parsed, quality checked and partitioned; no prior model fit or forecast score identified for these three markets",
        "source_limitation": "author monthly-reference anomaly cleaning/replacement can use future-within-month information; this is evaluation of the published cleaned benchmark, not a certified live-vintage deployment test",
        "fixed_primary_candidate": "liquidity_full",
        "prespecified_secondary_candidate": "allocation_readout from v1 development discovery",
        "primary_markets": ["BEIJING_MEAN", "TIANJIN_MEAN"],
        "separate_stress_market": "CHONGQING_MEAN (28 common H20-complete test origins)",
        "neural_families": FAMILIES, "classical_families": ["rw", "drift", "ridge_price", "ridge_full", "hgb_full"],
        "partition": "preserve previous 60/20/20 raw-row boundaries: n*3//5, then n//5, remainder",
        "label_maturity": "origin t>=64; each split's t+20 must be before that split's end; final refit all matured pre-test labels",
        "horizons_observation_steps": common.HORIZONS.tolist(), "lookback": common.LOOKBACK,
        "learning_rates": common.LRS, "selection_seed": 17, "refit_seeds": common.SEEDS,
        "maximum_epochs": common.MAX_EPOCHS, "minimum_epochs_before_stopping": common.MIN_EPOCHS,
        "patience": common.PATIENCE, "batch_size": common.BATCH_SIZE,
        "optimizer": "frozen v1 AdamW(weight_decay=1e-4), global gradient norm clip 5",
        "training_loss": "equal H1/H5/H20 native squared error divided by training target second moment",
        "selection": "past validation only selects lr and epoch including epoch0, then three seeds refit all matured pre-test labels for selected epoch count",
        "test_updates": "all fitted weights frozen over test; observed history updated at each origin",
        "primary_seed_summary": "mean individual-seed loss, then equal horizon and primary-market means of MSE/RW",
        "secondary_seed_summary": "three-prediction ensemble scores labeled separately",
        "energy": "same three public historical EIA arithmetic-change features, backward asof >=14 calendar days old; not historical-vintage certified",
        "volume_context": "same v2 causal prior-60-record normalization; 64 input tokens may use 60 earlier records for volume reference",
        "control_limitations": ["v2 full and readout start at equal parameter values but different predictions", "constrained and signed controls have unequal initial effective-coefficient derivative scales", "price-only Ridge excludes volume intentionally; ridge_full/GRU/LSTM/HGB receive the same six channels"],
        "statistics_plan_sha256": common.digest(stat_path),
        "source_sha256": {name: common.digest(Path(__file__).parent/name) for name in source_names},
        "claim_rule": "report all markets/horizons and fixed controls; no test-driven retry, candidate promotion or universal percentage threshold; test-conditioned inference cannot certify source-time cleaning causality",
    }
    common.write_json(out/"PROTOCOL_BEFORE_RUN.json", protocol)
    for market in markets:
        d = prepare_replication(market)
        common.write_json(out/f"{market}_{FOLD}"/"INPUT_MANIFEST.json", d["manifest"])
    return protocol


def install_hooks():
    common.prepare = prepare_replication
    common.make_model = make_replication_model


install_hooks()


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markets", nargs="+", choices=MARKETS, default=MARKETS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    DATA_ROOT = args.data_root.resolve()
    os.environ["CARBON_SCIENCEDB_DATA_DIR"] = str(DATA_ROOT)
    protocol = freeze_replication(args.out, args.markets)
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {}
        for market in args.markets:
            jobs[pool.submit(common.classical_job, market, FOLD, str(args.out))] = f"{market}/classical"
            for family in FAMILIES:
                jobs[pool.submit(common.neural_job, market, FOLD, family, str(args.out))] = f"{market}/{family}"
        for fut in as_completed(jobs):
            result = fut.result()
            rows = result if isinstance(result, list) else [result]
            summaries.extend(rows)
            print(json.dumps({"completed": jobs[fut], "scores": [{k:r[k] for k in ["family", "mean_seed_mse_over_rw", "ensemble_mse_over_rw"]} for r in rows]}), flush=True)
            common.write_json(args.out/"PROGRESS_SUMMARIES.json", summaries)
    common.write_json(args.out/"ALL_SUMMARIES.json", summaries)
    common.write_json(args.out/"EXECUTION_COMPLETE.json", {
        "status": "completed", "jobs": len(jobs), "summary_rows": len(summaries),
        "source_hashes_unchanged": protocol["source_sha256"] == {name: common.digest(Path(__file__).parent/name) for name in protocol["source_sha256"]},
        "statistics_plan_hash_unchanged": protocol["statistics_plan_sha256"] == common.digest(common.STUDY/"audits/V3_STATISTICAL_AUDIT_PLAN.json"),
    })


if __name__ == "__main__":
    main()
