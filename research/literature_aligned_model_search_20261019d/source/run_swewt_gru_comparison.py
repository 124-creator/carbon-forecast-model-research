"""Controlled multi-step adaptation of the published SWEWT-GRU structure.

Added after partial v3 results were inspected, not part of v3 registration.
Every EWT uses only the 64 observed price levels ending at its own origin.
The original paper's tuned single-step experiment is not numerically reproduced.
This separate runner imports, but never edits, the frozen v1-v3 sources.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import inspect
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

import run_new_market_replication as replication
from literature_diagnostic import decompose
from ewtpy import EWT1D


common = replication.common
WINDOW, SEQUENCE, MODES = 64, 17, 3
FAMILIES = ["swewt_gru_price", "swewt_gru_full"]
DEFAULT_REFERENCE = Path("/tmp/carbon-literature-aligned-runs-20261019d/new_market_v3")
PAPER = {
    "title": "Carbon Price Forecasting Using Optimized Sliding Window Empirical Wavelet Transform and Gated Recurrent Unit Network to Mitigate Data Leakage",
    "doi": "10.3390/en17174358", "venue": "Energies 17(17), 4358 (2024)",
    "official_url": "https://www.mdpi.com/1996-1073/17/17/4358",
    "publisher_provided_full_text": "https://www.researchgate.net/publication/383685619_Carbon_Price_Forecasting_Using_Optimized_Sliding_Window_Empirical_Wavelet_Transform_and_Gated_Recurrent_Unit_Network_to_Mitigate_Data_Leakage",
}


def price_sequence_at_origin(price, origin, target_scale):
    """Raw and EWT level features, without ever supplying future rows to EWT."""
    p = np.asarray(price, dtype=float)
    origin = int(origin)
    if p.ndim != 1 or not WINDOW-1 <= origin < len(p):
        raise ValueError("An origin needs 64 observed price levels")
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Use the positive target scale fixed by the v3 training prefix")
    observed = p[origin-WINDOW+1:origin+1].copy()
    if not np.isfinite(observed).all():
        raise ValueError("Nonfinite observed price window")
    bands = decompose(observed)[-SEQUENCE:].copy()
    raw = observed[-SEQUENCE:]-p[origin]
    # Only the low band carries the price-level offset; higher bands retain
    # their component values. No horizon-specific target is used to center.
    bands[:, 0] -= p[origin]
    return (np.column_stack((raw,bands))/target_scale).astype("float32")


def build_sequences(price, origins, target_scale, normalized_v3_windows):
    x = np.asarray(normalized_v3_windows)
    if x.ndim != 3 or x.shape != (len(origins),WINDOW,6):
        raise ValueError("Expected the unchanged v3 64-by-6 input windows")
    price_features = np.stack([
        price_sequence_at_origin(price,origin,target_scale) for origin in origins
    ])
    # The three energy columns already use their own fixed training RMS;
    # volume surprise/mask already use the preserved causal volume rule.
    auxiliary = x[:, -SEQUENCE:, 1:6].astype("float32",copy=True)
    packed = np.concatenate((price_features,auxiliary),axis=-1)
    if packed.shape != (len(origins),SEQUENCE,9) or not np.isfinite(packed).all():
        raise ValueError("Invalid past-window SWEWT sequence")
    return packed


def prepare_swewt(market, fold=replication.FOLD):
    d = replication.prepare_replication(market,fold)
    base_manifest = copy.deepcopy(d["manifest"])
    d["x"] = build_sequences(d["price"],d["origins"],d["scale"][0],d["x"])
    d["manifest"] = {
        **base_manifest,
        "swewt_adaptation": {
            "classification": "controlled multi-step adaptation, not author-code/numerical reproduction",
            "paper_doi": PAPER["doi"], "ewt_window_price_levels": WINDOW,
            "retained_sequence_length": SEQUENCE, "modes": MODES,
            "ewt_settings": {"log":0,"detect":"locmax","completion":1,"reg":"average","lengthFilter":10,"sigmaFilter":5},
            "decomposition_information": "separate EWT of price[t-63:t+1] at every origin; no full-series decomposition, future rows, or future labels",
            "columns": ["raw_level_minus_origin_price","low_EWT_band_minus_origin_price","EWT_band_2","EWT_band_3","brent_change","wti_change","henryhub_change","volume_surprise","volume_valid"],
            "normalization": "first four channels divided by unchanged training-prefix target RMS; last five retain v3 normalization",
            "price_arm_input": "only first four channels enter its GRU; no unused exogenous GRU weights",
            "full_arm_input": "all nine channels enter its GRU; only the last 17 v3 energy/volume context tokens",
            "information_span": "price bands use 64 levels; 17 volume tokens can use up to 60 preceding records for their reference; auxiliary sequence is shorter than the v3 64-token recurrent baselines",
            "original_v3_manifest": base_manifest,
        },
    }
    return d


class SWEWTGRU(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        if input_channels not in (4,9):
            raise ValueError("The two fixed arms require four or nine channels")
        self.input_channels = int(input_channels)
        self.encoder = nn.GRU(self.input_channels,32,batch_first=True)
        self.readout = nn.Linear(32,20)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self,x):
        if x.ndim != 3 or x.shape[1] != SEQUENCE:
            raise ValueError("SWEWT-GRU expects a 17-token sequence")
        if self.input_channels == 4:
            if x.shape[-1] not in (4,9):
                raise ValueError("Price arm needs four features or their nine-channel packed view")
            x = x[..., :4]
        elif x.shape[-1] != 9:
            raise ValueError("Full arm needs all nine features")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Nonfinite SWEWT-GRU input")
        z,_ = self.encoder(x)
        return self.readout(z[:,-1])


def make_swewt_model(family,seed):
    if family not in FAMILIES:
        raise ValueError(f"Unsupported literature comparison arm: {family}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return SWEWTGRU(4 if family == "swewt_gru_price" else 9)


def install_hooks(data_root=None):
    if data_root is not None:
        replication.DATA_ROOT = Path(data_root)
    common.prepare = prepare_swewt
    common.make_model = make_swewt_model


def dependency_receipt():
    module_path=Path(inspect.getsourcefile(EWT1D)).resolve()
    return {
        "python_executable":sys.executable,"python":platform.python_version(),
        "numpy":np.__version__,"torch":torch.__version__,
        "ewtpy_version":importlib.metadata.version("ewtpy"),
        "ewtpy_module_path":str(module_path),"ewtpy_module_sha256":common.digest(module_path),
    }


def freeze_comparison(out,markets,reference_v3_run):
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Preserve earlier output; this comparison needs a new empty directory")
    reference_path=Path(reference_v3_run)/"PROTOCOL_BEFORE_RUN.json"
    reference=json.loads(reference_path.read_text())
    prior_hashes=reference["source_sha256"]
    for name,value in prior_hashes.items():
        assert common.digest(Path(__file__).parent/name) == value, "Frozen v1-v3 source changed"
    source_names=[*prior_hashes,"literature_diagnostic.py",Path(__file__).name]
    source_hashes={name:common.digest(Path(__file__).parent/name) for name in source_names}
    assert set(markets).issubset(set(reference["markets"]))
    protocol={
        "stage":"additional_published_structure_control_after_partial_v3_results",
        "created_unix":time.time(),"paper":PAPER,
        "registration_status":"added after at least Beijing v3 scores were inspected; not among original v3 registered baselines",
        "purpose":"descriptive same-target comparison with a published past-window decomposition plus recurrent forecasting structure; no new model candidate or p-value search",
        "markets":markets,"fold":replication.FOLD,"families":FAMILIES,
        "target":reference["target"],"source_limitation":reference["source_limitation"],
        "primary_markets":reference["primary_markets"],"separate_stress_market":reference["separate_stress_market"],
        "reference_v3_protocol_path":str(reference_path),"reference_v3_protocol_sha256":common.digest(reference_path),
        "reference_v3_manifest_sha256":{
            market:common.digest(Path(reference_v3_run)/f"{market}_{replication.FOLD}"/"INPUT_MANIFEST.json")
            for market in markets
        },
        "source_sha256":source_hashes,"runtime":dependency_receipt(),
        "window_price_levels":WINDOW,"sequence_length":SEQUENCE,"ewt_modes":MODES,
        "decomposition":"ewtpy EWT1D called independently on only the 64 observed levels ending at each origin; exact frozen settings from literature_diagnostic.decompose",
        "input_channels":{"swewt_gru_price":4,"swewt_gru_full":9},
        "nominal_parameter_counts":{"swewt_gru_price":4308,"swewt_gru_full":4788},
        "potentially_data_active_parameters_at_supervised_horizons":{"swewt_gru_price":3747,"swewt_gru_full":4227},
        "readout":"20 zero-initialized linear outputs; only rows h1,h5,h20 receive the common loss, other 17 rows are not claimed to be supervised forecasts",
        "normalization":"raw price and low EWT band subtract origin price; all four price channels divide fixed v3 target RMS; energy/volume retain v3 transforms",
        "horizons_observation_steps":common.HORIZONS.tolist(),
        "learning_rates":common.LRS,"selection_seed":17,"refit_seeds":common.SEEDS,
        "maximum_epochs":common.MAX_EPOCHS,"minimum_epochs_before_stopping":common.MIN_EPOCHS,
        "patience":common.PATIENCE,"batch_size":common.BATCH_SIZE,
        "optimizer":reference["optimizer"],"training_loss":reference["training_loss"],
        "selection":reference["selection"],"label_maturity":reference["label_maturity"],
        "initialization":"GRU initial weights use the paired fixed seed; zero readout gives RW before any update",
        "primary_seed_summary":reference["primary_seed_summary"],
        "secondary_seed_summary":reference["secondary_seed_summary"],
        "planned_neural_tasks":len(markets)*len(FAMILIES)*(len(common.LRS)+len(common.SEEDS)),
        "comparison_scope":"do not rerun v3 controls; compare on identical existing targets/origins; price arm omits exogenous information intentionally",
        "departures_from_original_paper":[
            "three ScienceDB transaction-mean benchmark markets and preserved 60/20/20 raw-row boundaries instead of the paper's single-market 70/10/20 experiment",
            "fixed window64 and three bands within the published search ranges, rather than TPE searching window24-64 and two-four bands",
            "single GRU32 and a direct 20-row head with supervised horizons1/5/20, rather than the paper's tuned single-step forecast",
            "origin-centered level features with fixed training RMS and original-price horizon-risk loss, rather than the paper's train-minmax level formulation",
            "two fixed learning rates, at most80 epochs/patience12 and three specified refit seeds, rather than paper TPE runs10/maxepochs1000/patience100",
            "full arm adds permitted energy and volume context; these inputs were not in the original univariate paper",
            "the ewtpy implementation and fixed EWT options are recorded; not an author-code or numerical reproduction",
        ],
        "statistics":"no new p-values or post-result choice between these two arms; retain price and full outputs separately",
    }
    common.write_json(out/"PROTOCOL_BEFORE_RUN.json",protocol)
    for market in markets:
        d=prepare_swewt(market)
        original_path=Path(reference_v3_run)/f"{market}_{replication.FOLD}"/"INPUT_MANIFEST.json"
        assert json.loads(original_path.read_text()) == d["manifest"]["swewt_adaptation"]["original_v3_manifest"]
        common.write_json(out/f"{market}_{replication.FOLD}"/"INPUT_MANIFEST.json",d["manifest"])
    return protocol


# Worker imports establish the same process-local hooks. Frozen modules on
# disk and prior experiment artifacts are never changed.
install_hooks()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--markets",nargs="+",choices=replication.MARKETS,default=replication.MARKETS)
    parser.add_argument("--data-root",type=Path,default=replication.DATA_ROOT)
    parser.add_argument("--reference-v3-run",type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    data_root=args.data_root.resolve()
    os.environ["CARBON_SCIENCEDB_DATA_DIR"]=str(data_root)
    install_hooks(data_root)
    protocol=freeze_comparison(args.out,args.markets,args.reference_v3_run)
    summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=install_hooks,initargs=(str(data_root),)) as pool:
        jobs={pool.submit(common.neural_job,market,replication.FOLD,family,str(args.out)):f"{market}/{family}"
              for market in args.markets for family in FAMILIES}
        for future in as_completed(jobs):
            result=future.result()
            summaries.append(result)
            print(json.dumps({"completed":jobs[future],"score":{k:result[k] for k in ["family","mean_seed_mse_over_rw","ensemble_mse_over_rw"]}}),flush=True)
            common.write_json(args.out/"PROGRESS_SUMMARIES.json",summaries)
    unchanged=all(common.digest(Path(__file__).parent/name) == value for name,value in protocol["source_sha256"].items())
    dependency_unchanged=dependency_receipt()["ewtpy_module_sha256"] == protocol["runtime"]["ewtpy_module_sha256"]
    common.write_json(args.out/"ALL_SUMMARIES.json",summaries)
    common.write_json(args.out/"EXECUTION_COMPLETE.json",{
        "status":"completed","jobs":len(jobs),"summary_rows":len(summaries),
        "source_hashes_unchanged":unchanged,"ewtpy_module_hash_unchanged":dependency_unchanged,
    })
    assert unchanged and dependency_unchanged


if __name__ == "__main__":
    main()
