"""Recorded development extension using the downloaded author volume columns.

This imports the frozen v1 runner and explicitly replaces its model/data hooks
in this process only.  It neither edits that source nor changes any v1 result.
The shared optimizer, split, maturity, scaler and scoring code remain identical.
Linux fork workers inherit these hooks; a guarded initializer also installs
them if this module is imported by a spawn-based worker.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_model_search as common


BASE_PREPARE = common.prepare
BASE_FREEZE = common.freeze
AUXILIARY_ROOT = Path(os.environ.get("CARBON_VOLUME_AUX_DIR", "/tmp/carbon-literature-data-audit-20261019d/prepared"))
FAMILIES = ["liquidity_full", "liquidity_readout", "shock_size_only",
            "liquidity_unconstrained", "gru", "lstm"]


def prepare_liquidity(market, year):
    d = BASE_PREPARE(market, year)
    path = AUXILIARY_ROOT/f"{market}_auxiliary.csv"
    aux = pd.read_csv(path)
    adates = pd.to_datetime(aux.date).to_numpy(dtype="datetime64[D]")
    assert np.array_equal(adates, d["dates"]), "Auxiliary dates must equal the canonical price dates"
    valid = (aux.valid_volume.to_numpy() == 1) & np.isfinite(aux.volume.to_numpy(float))
    assert (aux.volume.to_numpy(float)[valid] >= 0).all()
    logs = np.full(len(aux), np.nan)
    logs[valid] = np.log1p(aux.volume.to_numpy(float)[valid])
    context = np.zeros((len(aux), 2), dtype="float32")
    for t in range(len(aux)):
        past = logs[max(0,t-60):t]
        past = past[np.isfinite(past)]
        if valid[t] and len(past) >= 10:
            context[t, 0] = np.clip((logs[t]-np.median(past))/max(float(np.std(past)), .25), -5, 5)
            context[t, 1] = 1.0
    windows = np.stack([context[t-common.LOOKBACK+1:t+1] for t in d["origins"]])
    d["x"] = np.concatenate([d["x"], windows], axis=-1)
    d["manifest"]["volume_source"] = {
        "path": str(path), "sha256": common.digest(path),
        "rows": len(aux), "valid_reported_volume_rows": int(valid.sum()),
        "usable_context_rows": int(context[:,1].sum()),
        "rule": "current log1p volume minus median of valid volumes in prior 60 target records; divide prior standard deviation floored .25, clip [-5,5]; require >=10 past valid rows",
        "missing_rule": "context=(0,0) means absent information; never asserts volume itself equals zero",
        "information": "author-reported current-day Volume assumed available after that day's report; exact publication vintage/unit not certified",
        "input_information_span": "64 return tokens; each volume token uses up to 60 earlier target records for its reference",
    }
    return d


class DirectVolumeRecurrent(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.encoder = (nn.GRU if kind == "gru" else nn.LSTM)(6,32,batch_first=True)
        self.readout = nn.Linear(32,20)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self,x):
        z,_ = self.encoder(x)
        return self.readout(z[:,-1])


def make_liquidity_model(family, seed):
    from liquidity_models import LiquidityAllocationForecaster
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if family in ["gru","lstm"]:
        return DirectVolumeRecurrent(family)
    return LiquidityAllocationForecaster(n_exog=3,horizon=20,variant=family)


def freeze_liquidity(out, markets, years):
    protocol = BASE_FREEZE(out, markets, years)
    protocol.update({
        "study_version": "liquidity_extension_after_partial_v1_development_results",
        "relationship_to_v1": "same price targets, folds and energy features; add downloaded author volume; new constrained gate and matched readout control; do not overwrite or pool v1 test scores",
        "reason": "small-data MLP allocation was unstable in the first GZ/HB development cells; test observed trading volume and a parsimonious sign-constrained hypothesis instead",
        "volume_hypothesis": "at fixed volume larger absolute innovation is more transient; at fixed innovation positive volume surprise is more permanent; these signs are testable restrictions, not established economic facts",
        "critical_controls": ["liquidity_readout (same capacity)", "shock_size_only (remove volume from gate)",
                              "liquidity_unconstrained (remove economic sign restrictions)",
                              "v1 allocation_constant (same target and energy information, volume ignored)",
                              "GRU/LSTM/Ridge/HGB with the same added volume inputs"],
        "frozen_prior_diagnostic": "also score the four compact gate models at their fixed initial coefficients without fitting, to separate a useful hand-specified prior from gains learned from data; secondary diagnostic",
        "volume_source_directory": str(AUXILIARY_ROOT),
        "source_sha256": {**protocol["source_sha256"],
                          "run_liquidity_extension.py":common.digest(Path(__file__)),
                          "liquidity_models.py":common.digest(Path(__file__).with_name("liquidity_models.py"))},
    })
    common.write_json(Path(out)/"PROTOCOL_BEFORE_RUN.json",protocol)
    return protocol


def install_hooks():
    common.prepare = prepare_liquidity
    common.make_model = make_liquidity_model
    common.freeze = freeze_liquidity
    common.MARKETS = ["GUANGZHOU","HUBEI"]
    common.FAMILIES = FAMILIES


# Worker imports run this too; the frozen base module itself remains unchanged.
install_hooks()


if __name__ == "__main__":
    # The worker-compatible default is the path in the source receipt.  Set a
    # different local directory via this explicit optional CLI argument.
    import sys
    if "--auxiliary-root" in sys.argv:
        at = sys.argv.index("--auxiliary-root")
        AUXILIARY_ROOT = Path(sys.argv[at+1]).resolve()
        os.environ["CARBON_VOLUME_AUX_DIR"] = str(AUXILIARY_ROOT)
        del sys.argv[at:at+2]
    common.main()
