"""Frozen-budget GAS-Norm / EWMA / Global probability-forecast comparison.

This runner uses only the four pre-existing development markets, h=5/20.
It never opens the additional Shenzhen panels and never downloads data.
--print-source-hash is read-only and reads no market file.
--prepare-only performs the 22 static GAS fits and fixed control construction.
--train-only performs 198 neural fits, with validation including epoch zero.
--aggregate-only produces value-free public tables from private predictions.
Every operational stage requires the predeclared source fingerprint explicitly.
All checkpoints and per-origin arrays remain outside the Git repository.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F

from gas_filter import fit_gas_normalizer, filter_gas, forecast_gas
from mixture_metrics import score_mixture, score_atoms, torch_gaussian_mixture_crps


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
MARKETS = ("EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI")
PANELS = tuple(f"{m}_{y}" for m in MARKETS for y in (2021,2022,2023) if (m,y) != ("SHANGHAI",2021))
HORIZONS = (5,20)
NORMALIZERS = ("global", "ewma", "gas")
SEEDS = (17,41,89)
ROLES = ("train", "validation", "test")
EPOCHS = 40
BATCH = 64
WEIGHTS0 = np.array([.45,.4,.15], dtype=float)
SIGMA0 = np.array([.5,1.,2.], dtype=float)
SIGMA0 /= np.sqrt(np.dot(WEIGHTS0, SIGMA0**2))
SIGMA_FLOOR = 1e-4
RAW_SCALE0 = np.log(np.expm1(SIGMA0-SIGMA_FLOOR))
KNOWN_SOURCE_SHA = {
    "EUA_LONG":"142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f",
    "GUANGZHOU":"8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051",
    "HUBEI":"04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797",
    "SHANGHAI":"425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1",
}
SOURCE_FILES = ("source/gas_filter.py", "source/mixture_metrics.py", "source/run_gas_benchmark.py",
                "data_sources/prepare_data.py", "provenance/PROTOCOL.json",
                "provenance/GAS_IMPLEMENTATION_PRE_FIT.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    files = {name: sha(ROOT/name) for name in SOURCE_FILES}
    fingerprint = hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"files": files, "fingerprint": fingerprint}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".writing")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".writing")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def private_root(path):
    result = Path(path).resolve()
    if result == REPO or REPO in result.parents:
        raise ValueError("Private arrays/checkpoints cannot be saved in the Git repository")
    return result


def stable_average(values, axis=0):
    values = np.asarray(values)
    first = np.take(values,0,axis=axis)
    return first + np.mean(values-np.expand_dims(first,axis),axis=axis)


def future_target_process(returns, horizon):
    returns = np.asarray(returns,dtype=float)
    if returns.ndim != 1 or len(returns) <= horizon or returns[0] != 0 or not np.isfinite(returns).all():
        raise ValueError("Expected finite returns with the documented index-zero sentinel")
    cumulative = np.cumsum(returns)
    # First entry is x_h, built from actual returns 1,...,h.
    return cumulative[horizon:]-cumulative[:-horizon]


def simple_normalizer(values, fit_end_exclusive, name):
    values = np.asarray(values,dtype=float)
    prefix = values[:fit_end_exclusive]
    mean0, sd0 = float(prefix.mean()), float(prefix.std(ddof=0))
    if sd0 <= 1e-12:
        raise ValueError("Constant mature training process")
    if name == "global":
        z = (values-mean0)/sd0
        return z, np.full(len(values),mean0), np.full(len(values),sd0), {"mean":mean0,"std":sd0}
    if name != "ewma":
        raise ValueError("Unknown simple normalizer")
    floor2 = (sd0*1e-4)**2
    mean, variance = mean0, sd0**2
    z, forecast_mean, forecast_sd = [np.empty(len(values)) for _ in range(3)]
    for i,value in enumerate(values):
        z[i] = (value-mean)/math.sqrt(variance)
        residual = value-mean
        mean = .94*mean+.06*value
        variance = max(.94*variance+.06*residual**2,floor2)
        forecast_mean[i], forecast_sd[i] = mean, math.sqrt(variance)
    return z, forecast_mean, forecast_sd, {"initial_mean":mean0,"initial_std":sd0,
        "decay":.94,"std_floor":sd0*1e-4,"future_rule":"carry_last_next_prior"}


def read_base(path):
    receipt = json.loads(Path(path).with_suffix(".json").read_text())
    panel = Path(path).stem
    if panel not in PANELS or receipt.get("cell") != panel:
        raise ValueError("GAS benchmark accepts only the eleven fixed development panels and matching receipts")
    market, year_text = panel.rsplit("_",1)
    year = int(year_text)
    if market not in KNOWN_SOURCE_SHA or receipt["prepared_sha256"] != sha(path):
        raise ValueError("Prepared file hash or market identity mismatch")
    if receipt["source"]["sha256"] != KNOWN_SOURCE_SHA[market]:
        raise ValueError("Source bytes differ from the fixed development source")
    with np.load(path,allow_pickle=False) as f:
        dates = f["all_dates"].astype("U10")
        if len(dates) < 1 or np.any(dates >= "2024-01-01") or np.any(dates[1:] <= dates[:-1]):
            raise ValueError("Reserved/nonincreasing dates rejected before return-array access")
        if str(f["source_sha256"].item()) != KNOWN_SOURCE_SHA[market]:
            raise ValueError("NPZ source hash and fixed source receipt disagree")
        if tuple(f["horizons"].tolist()) != (1,5,20):
            raise ValueError("Unexpected prepared horizon columns")
        # Input validation added during pre-fit review: recompute the exact
        # protocol from dates, not merely a weaker nonoverlap inequality.
        same_year = np.char.startswith(dates, f"{year}-")
        in_year = np.flatnonzero(same_year)
        if not len(in_year):
            raise ValueError("The prescribed annual panel has no dates")
        first_test = int(in_year[0])
        expected = {
            "train": np.arange(39, first_test-166, dtype=np.int64),
            "validation": np.arange(first_test-146, first_test-20, dtype=np.int64),
            "test": np.flatnonzero(same_year[:-20] & same_year[20:]),
        }
        if len(expected["train"]) < 128 or len(expected["validation"]) != 126 or not len(expected["test"]):
            raise ValueError("Fixed sample-count rule is not met")
        for role, ids in expected.items():
            actual = f[f"{role}_indices"]
            if not np.issubdtype(actual.dtype, np.integer):
                raise ValueError("Prepared origin indices must be integer-valued")
            np.testing.assert_array_equal(actual, ids)
            np.testing.assert_array_equal(f[f"{role}_dates"], dates[ids])
            target_indices = ids[:,None] + np.array([1,5,20])
            np.testing.assert_array_equal(f[f"{role}_target_indices"], target_indices)
            np.testing.assert_array_equal(f[f"{role}_target_dates"], dates[target_indices])
            if receipt["roles"][role]["count"] != len(ids):
                raise ValueError("Producer role counts disagree with the fixed split")
        train, val, test = [expected[role] for role in ROLES]
        label_end = int(f["train_label_end"])
        if not (label_end == train[-1]+20 and label_end < val[0] and val[-1]+20 < test[0]):
            raise ValueError("A training or validation endpoint has not matured")
        result = {key:f[key] for key in f.files}
    return result


def head_distribution(raw):
    raw = np.asarray(raw,dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError("Expected nine raw GMM outputs")
    w = softmax(raw[:,:3]+np.log(WEIGHTS0),axis=1)
    m = raw[:,3:6]
    s = np.logaddexp(0.,raw[:,6:9]+RAW_SCALE0)+SIGMA_FLOOR
    return w,m,s


def distribution_in_scoring_units(raw, loc, scale, unit):
    w,m,s = head_distribution(raw)
    return w,(loc[:,None]+scale[:,None]*m)/unit[:,None],scale[:,None]*s/unit[:,None]


def prediction_arrays(prepared, role, w, m, s):
    scores = score_mixture(prepared[f"{role}_y"],w,m,s,
                          origin_price=prepared[f"{role}_origin_price"],unit=prepared[f"{role}_unit"],
                          target_price=prepared[f"{role}_target_price"])
    return {"weights":w,"means":m,"sigmas":s,
            "origin":prepared[f"{role}_indices"],"date":prepared[f"{role}_dates"],
            "target_index":prepared[f"{role}_target_indices"],"target_date":prepared[f"{role}_target_dates"],
            "truth":prepared[f"{role}_y"],"unit":prepared[f"{role}_unit"],
            "origin_price":prepared[f"{role}_origin_price"],"target_price":prepared[f"{role}_target_price"],
            **{f"score_{name}":value for name,value in scores.items()}}


def public_scores(arrays):
    return {key[6:]:float(np.mean(value)) for key,value in arrays.items() if key.startswith("score_")}


def build_prepared(base, horizon):
    x = future_target_process(base["all_returns"],horizon)
    endpoint = int(base["train_label_end"])
    fit_count = endpoint-horizon+1
    gas_fit = fit_gas_normalizer(x[:fit_count])
    gas_hist = filter_gas(x,gas_fit)
    gas_loc,gas_sd = forecast_gas(gas_fit,gas_hist,horizon=horizon)
    normalized = {"gas":(gas_hist.normalized,gas_loc,gas_sd,gas_fit.as_dict())}
    normalized.update({name:simple_normalizer(x,fit_count,name) for name in ("global","ewma")})
    column = int(np.flatnonzero(base["horizons"] == horizon)[0])
    output = {"horizon":np.array(horizon),"train_label_end":np.array(endpoint),
              "source_sha256":base["source_sha256"]}
    normalizer_metadata = {}
    for role in ROLES:
        ids = base[f"{role}_indices"]
        output.update({f"{role}_{key}":base[f"{role}_{key}"] for key in
                       ("indices","dates","origin_price")})
        for key in ("y","unit","target_price","target_indices","target_dates"):
            output[f"{role}_{key}"] = base[f"{role}_{key}"][:,column]
        # x[j] has maturity j+h. Thus x[origin] is this origin's forward target.
        expected = x[ids]
        np.testing.assert_allclose(expected,output[f"{role}_y"]*output[f"{role}_unit"],atol=1e-11,rtol=1e-9)
        at = ids-horizon
        context = at[:,None]+np.arange(-19,1)
        if context.min() < 0:
            raise ValueError("Target-process context uses an immature or sentinel observation")
        for name,(z,loc,sd,meta) in normalized.items():
            output[f"{role}_X_{name}"] = z[context,None].astype(np.float32)
            output[f"{role}_loc_{name}"] = loc[at]
            output[f"{role}_scale_{name}"] = sd[at]
            normalizer_metadata[name] = meta
    # Fixed 101-atom empirical forecast of the normalized future residual.
    levels = (np.arange(101)+.5)/101
    for name in NORMALIZERS:
        residual = (output["train_y"]*output["train_unit"]-output[f"train_loc_{name}"])/output[f"train_scale_{name}"]
        output[f"empirical_residual_atoms_{name}"] = np.quantile(residual,levels,method="linear")
    return output,normalizer_metadata


def prepare_one(base_path,horizon,local_dir,fingerprint):
    started = time.time()
    base_path,local_dir = Path(base_path),Path(local_dir)
    cell = f"{base_path.stem}_h{horizon}"
    out = local_dir/"prepared"/f"{cell}.npz"
    meta_path = out.with_suffix(".json")
    attempt_path = local_dir/"attempts"/f"prepare_{cell}.json"
    base_sha = sha(base_path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["source_fingerprint"] != fingerprint or meta["base_prepared_sha256"] != base_sha or meta["prepared_sha256"] != sha(out):
            raise ValueError("A stale GAS prepared artifact cannot be resumed")
        with np.load(out,allow_pickle=False) as loaded:
            arrays = {key:loaded[key] for key in loaded.files}
        save_all_controls(local_dir,cell,arrays,fingerprint)
        return meta["public_receipt"]
    if out.exists() or attempt_path.exists():
        raise RuntimeError("Incomplete static-fit job exists; require an explicitly recorded recovery, not silent refitting")
    base = read_base(base_path)
    write_json(attempt_path,{"cell":cell,"stage":"static_fit_started","source_fingerprint":fingerprint,
                            "base_prepared_sha256":base_sha,"utc":datetime.now(timezone.utc).isoformat()})
    arrays,norm = build_prepared(base,horizon)
    write_npz(out,**arrays)
    fit_meta = norm["gas"]["metadata"]
    public = {"cell":cell,"horizon":horizon,"panel":base_path.stem,"source_fingerprint":fingerprint,
              "base_prepared_sha256":base_sha,"prepared_sha256":sha(out),
              "roles":{role:{"n":len(arrays[f"{role}_indices"]),
                             "first_origin":int(arrays[f"{role}_indices"][0]),
                             "last_origin":int(arrays[f"{role}_indices"][-1])} for role in ROLES},
              "gas_train_n":fit_meta["train_observations"],"gas_nfev":fit_meta["nfev"],
              "gas_optimizer_success":fit_meta["optimizer_success"],
              "gas_optimizer_message":fit_meta["optimizer_message"],
              "normalizer_static_fits":1,"seconds":time.time()-started}
    write_json(meta_path,{"source_fingerprint":fingerprint,"base_prepared_sha256":base_sha,
                         "prepared_sha256":sha(out),"normalizers":norm,"public_receipt":public})
    save_all_controls(local_dir,cell,arrays,fingerprint)
    return public


def save_all_controls(local_dir,cell,arrays,fingerprint):
    # Seven fixed controls per cell: 3 no-neural, 3 empirical, common EWMA N(0,1).
    for name in NORMALIZERS:
        role = "test"; n = len(arrays["test_y"])
        w,m,s = distribution_in_scoring_units(np.zeros((n,9)),arrays[f"test_loc_{name}"],
                                             arrays[f"test_scale_{name}"],arrays["test_unit"])
        forecast = prediction_arrays(arrays,role,w,m,s)
        save_control(local_dir,cell,f"{name}_no_neural",forecast,fingerprint)
        atoms = (arrays[f"test_loc_{name}"][:,None]+arrays[f"test_scale_{name}"][:,None]*
                 arrays[f"empirical_residual_atoms_{name}"][None,:])/arrays["test_unit"][:,None]
        scores = score_atoms(arrays["test_y"],atoms,origin_price=arrays["test_origin_price"],
                             unit=arrays["test_unit"],target_price=arrays["test_target_price"])
        empirical = {key:value for key,value in forecast.items() if not key.startswith("score_") and key not in ("weights","means","sigmas")}
        empirical.update({"atoms":atoms,**{f"score_{key}":value for key,value in scores.items()}})
        save_control(local_dir,cell,f"{name}_empirical101",empirical,fingerprint)
    daily = prediction_arrays(arrays,"test",np.ones((n,1)),np.zeros((n,1)),np.ones((n,1)))
    save_control(local_dir,cell,"daily_ewma_normal",daily,fingerprint)


def save_control(local_dir,cell,family,arrays,fingerprint):
    path = Path(local_dir)/"predictions"/cell/family/"seed_0.npz"
    if path.with_suffix(".json").exists():
        previous = json.loads(path.with_suffix(".json").read_text())
        if previous["source_fingerprint"] != fingerprint or previous["prediction_sha256"] != sha(path):
            raise ValueError("An existing control prediction changed")
        return
    write_npz(path,**arrays)
    write_json(path.with_suffix(".json"),{"cell":cell,"family":family,"seed":0,
        "source_fingerprint":fingerprint,"neural_fit":False,"prediction_sha256":sha(path),
        "density_NLL":"undefined_for_atoms" if "atoms" in arrays else "analytic_mixture",
        "metrics":public_scores(arrays)})


class GRUMixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1,16,num_layers=1,batch_first=True)
        self.head = nn.Linear(16,9)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self,x):
        _,hidden = self.gru(x)
        return self.head(hidden[-1])


def training_distribution(raw,loc,scale,unit):
    basew = torch.as_tensor(np.log(WEIGHTS0),dtype=raw.dtype,device=raw.device)
    bases = torch.as_tensor(RAW_SCALE0,dtype=raw.dtype,device=raw.device)
    weights = torch.softmax(raw[:,:3]+basew,dim=1)
    means = (loc[:,None]+scale[:,None]*raw[:,3:6])/unit[:,None]
    sigmas = scale[:,None]*(F.softplus(raw[:,6:9]+bases)+SIGMA_FLOOR)/unit[:,None]
    return weights,means,sigmas


def predict(model,arrays,role,name):
    model.eval()
    x = torch.from_numpy(arrays[f"{role}_X_{name}"])
    with torch.no_grad():
        raw = np.concatenate([model(part).numpy() for part in x.split(256)],axis=0)
    return distribution_in_scoring_units(raw,arrays[f"{role}_loc_{name}"],
                                         arrays[f"{role}_scale_{name}"],arrays[f"{role}_unit"])


def train_one(prepared_path,name,seed,local_dir,fingerprint):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    prepared_path,local_dir = Path(prepared_path),Path(local_dir)
    cell, family = prepared_path.stem,f"{name}_gru_gmm"
    out = local_dir/"predictions"/cell/family/f"seed_{seed}.npz"
    meta_path = out.with_suffix(".json")
    checkpoint = local_dir/"checkpoints"/cell/family/f"seed_{seed}.pt"
    attempt_path = local_dir/"attempts"/f"train_{cell}_{family}_seed_{seed}.json"
    prepared_sha = sha(prepared_path)
    if meta_path.exists():
        receipt = json.loads(meta_path.read_text())
        if receipt["source_fingerprint"] != fingerprint or receipt["prepared_sha256"] != prepared_sha:
            raise ValueError("A stale trained artifact cannot be resumed")
        if receipt["prediction_sha256"] != sha(out) or receipt["checkpoint_sha256"] != sha(checkpoint):
            raise ValueError("A resumed prediction/checkpoint hash changed")
        return receipt
    if out.exists() or checkpoint.exists() or attempt_path.exists():
        raise RuntimeError("Incomplete neural job exists; require an explicitly recorded recovery, not silent refitting")
    prep_meta = json.loads(prepared_path.with_suffix(".json").read_text())
    if prep_meta["source_fingerprint"] != fingerprint or prep_meta["prepared_sha256"] != prepared_sha:
        raise ValueError("Training preparation identity mismatch")
    with np.load(prepared_path,allow_pickle=False) as loaded:
        for role in ROLES:
            for suffix in ("dates","target_dates"):
                dates = loaded[f"{role}_{suffix}"].astype("U10")
                if np.any(dates >= "2024-01-01"):
                    raise ValueError("Reserved derived dates rejected before feature/target-array access")
        arrays = {key:loaded[key] for key in loaded.files}
    write_json(attempt_path,{"cell":cell,"family":family,"seed":seed,"stage":"neural_fit_started",
                            "source_fingerprint":fingerprint,"prepared_sha256":prepared_sha,
                            "utc":datetime.now(timezone.utc).isoformat()})
    model = GRUMixture()
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 1065:
        raise AssertionError("The shared GRU/GMM parameter budget changed")
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    train_x = torch.from_numpy(arrays[f"train_X_{name}"])
    train_y,loc,scale,unit = [torch.tensor(arrays[key],dtype=torch.float32) for key in
                            ("train_y",f"train_loc_{name}",f"train_scale_{name}","train_unit")]

    def validation():
        w,m,s = predict(model,arrays,"validation",name)
        from mixture_metrics import gaussian_mixture_crps
        return float(gaussian_mixture_crps(arrays["validation_y"],w,m,s).mean())

    best = validation(); best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch":0,"train_loss":None,"validation_normalized_CRPS":best}]
    updates = 0
    for epoch in range(1,EPOCHS+1):
        model.train()
        # Same ordered indices for every normalization arm at a given seed.
        order = np.random.default_rng(seed*1000+epoch).permutation(len(train_y))
        total = 0.0
        for start in range(0,len(order),BATCH):
            ids = torch.from_numpy(order[start:start+BATCH])
            optimizer.zero_grad(set_to_none=True)
            raw = model(train_x[ids])
            w,m,s = training_distribution(raw,loc[ids],scale[ids],unit[ids])
            loss = torch_gaussian_mixture_crps(train_y[ids],w,m,s).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite train CRPS; no replacement model fitted")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); updates += 1
            total += float(loss.detach())*len(ids)
        current = validation()
        history.append({"epoch":epoch,"train_loss":total/len(order),"validation_normalized_CRPS":current})
        if current < best:
            best,best_epoch,best_state = current,epoch,copy.deepcopy(model.state_dict())
    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    w,m,s = predict(model,arrays,"test",name)
    forecast = prediction_arrays(arrays,"test",w,m,s)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":best_state,"final_model_state":final_state,
                "final_optimizer_state":optimizer.state_dict(),"normalizer":prep_meta["normalizers"][name],
                "source_fingerprint":fingerprint,"prepared_sha256":prepared_sha,
                "selected_epoch":best_epoch,"seed":seed,"history":history},checkpoint)
    write_npz(out,**forecast)
    receipt = {"cell":cell,"panel":cell.rsplit("_h",1)[0],"horizon":int(arrays["horizon"]),
        "family":family,"normalizer":name,"seed":seed,"neural_fit":True,
        "source_fingerprint":fingerprint,"prepared_sha256":prepared_sha,
        "prediction_sha256":sha(out),"checkpoint_sha256":sha(checkpoint),
        "epochs_executed":EPOCHS,"selected_epoch":best_epoch,"gradient_updates":updates,
        "neural_parameters":parameter_count,"fitted_normalizer_parameters":10 if name=="gas" else 2,
        "best_validation_normalized_CRPS":best,"history":history,
        "train_loss_history_definition":"Sample-weighted pre-update minibatch losses along the epoch's changing parameter path; not a post-epoch full-training evaluation",
        "metrics":public_scores(forecast),
        "test_origins":len(arrays["test_y"]),"seconds":time.time()-started}
    write_json(meta_path,receipt)
    return receipt


def run_jobs(function,jobs,workers):
    receipts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function,*job):job for job in jobs}
        for completed in as_completed(futures):
            receipt = completed.result(); receipts.append(receipt)
            # No metric values are printed while models remain untrained.
            print(json.dumps({"completed":len(receipts),"total":len(jobs),"cell":receipt["cell"],
                              "family":receipt.get("family","static_prepare"),"seed":receipt.get("seed")}),flush=True)
    return sorted(receipts,key=lambda a:(a["cell"],a.get("family",""),a.get("seed",0)))


def aggregate(local_dir,public_dir,fingerprint):
    local_dir,public_dir = Path(local_dir),Path(public_dir)
    families = [f"{name}_gru_gmm" for name in NORMALIZERS]
    families += [f"{name}_{kind}" for name in NORMALIZERS for kind in ("no_neural","empirical101")]
    families += ["daily_ewma_normal"]
    fit_rows,cell_rows = [],[]
    cell_losses = {}
    expected_neural = 0
    for panel in PANELS:
        aligned = []
        common = None
        for h in HORIZONS:
            cell = f"{panel}_h{h}"; columns = []
            for family in families:
                seeds = SEEDS if family.endswith("gru_gmm") else (0,)
                seed_loss,seed_metrics = [],[]
                for seed in seeds:
                    path = local_dir/"predictions"/cell/family/f"seed_{seed}.npz"
                    receipt = json.loads(path.with_suffix(".json").read_text())
                    if receipt["source_fingerprint"] != fingerprint or receipt["prediction_sha256"] != sha(path):
                        raise ValueError("Aggregation forecast provenance mismatch")
                    with np.load(path,allow_pickle=False) as pred:
                        origin = pred["origin"]
                        if common is None: common = origin.copy()
                        np.testing.assert_array_equal(origin,common)
                        seed_loss.append(pred["score_normalized_CRPS"])
                    seed_metrics.append(receipt["metrics"])
                    expected_neural += int(receipt["neural_fit"])
                    row = {"cell":cell,"panel":panel,"horizon":h,"family":family,"seed":seed,
                           "neural_fit":receipt["neural_fit"],"n":len(origin),
                           "selected_epoch":receipt.get("selected_epoch"),
                           "seconds":receipt.get("seconds"),**receipt["metrics"]}
                    fit_rows.append(row)
                averaged = stable_average(seed_loss)
                columns.append(averaged)
                merged = {key:float(stable_average([a[key] for a in seed_metrics])) for key in seed_metrics[0]}
                cell_rows.append({"cell":cell,"panel":panel,"horizon":h,"family":family,
                                  "seed_count":len(seeds),"n":len(common),**merged})
            aligned.append(np.stack(columns,axis=1))
        cell_losses[panel] = np.stack(aligned,axis=1)  # origin x horizon x family
    if expected_neural != 198:
        raise AssertionError("The fixed 198-neural-fit plan is incomplete")
    public_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(fit_rows).to_csv(public_dir/"PER_FIT_METRICS.csv",index=False)
    pd.DataFrame(cell_rows).to_csv(public_dir/"CELL_METRICS.csv",index=False)
    pairs = [("gas_gru_gmm","ewma_gru_gmm"),("gas_gru_gmm","global_gru_gmm"),
             ("ewma_gru_gmm","global_gru_gmm")]
    for name in NORMALIZERS:
        pairs.extend((f"{name}_gru_gmm",f"{name}_{kind}") for kind in ("no_neural","empirical101"))
        pairs.append((f"{name}_gru_gmm","daily_ewma_normal"))
    rng = np.random.default_rng(20261019)
    bootstrap = np.zeros((1999,len(pairs)))
    point = [[] for _ in pairs]
    for panel in PANELS:
        loss = cell_losses[panel]
        n = len(loss); means = loss.mean(axis=0)
        for j,(a,b) in enumerate(pairs):
            point[j].extend((means[:,families.index(a)]/means[:,families.index(b)]-1).tolist())
        blocks = math.ceil(n/40)
        for begin in range(0,1999,100):
            count = min(100,1999-begin)
            starts = rng.integers(0,n,size=(count,blocks))
            indices = ((starts[:,:,None]+np.arange(40))%n).reshape(count,-1)[:,:n]
            sampled = loss[indices].mean(axis=1)
            for j,(a,b) in enumerate(pairs):
                bootstrap[begin:begin+count,j] += np.sum(sampled[:,:,families.index(a)]/
                    sampled[:,:,families.index(b)]-1,axis=1)/(len(PANELS)*len(HORIZONS))
    comparisons = []
    for j,(a,b) in enumerate(pairs):
        values = np.array(point[j]); ci = np.quantile(bootstrap[:,j],[.025,.975])
        comparisons.append({"candidate":a,"baseline":b,"cells":22,
            "mean_cell_relative_CRPS":float(values.mean()),"strictly_better_cells":int(np.sum(values<0)),
            "ci95_lower":float(ci[0]),"ci95_upper":float(ci[1])})
    pd.DataFrame(comparisons).to_csv(public_dir/"PAIRED_COMPARISONS.csv",index=False)
    selected = [r["selected_epoch"] for r in fit_rows if r["neural_fit"]]
    summary = {"source_fingerprint":fingerprint,"neural_fits":198,"static_gas_fits":22,
        "deterministic_control_forecasts":154,"epoch_zero_selections":int(np.sum(np.array(selected)==0)),
        "all_epochs_executed":40,"panels":list(PANELS),"horizons":list(HORIZONS),
        "primary_metric":"equal-cell seed-mean normalized CRPS",
        "bootstrap":{"kind":"paired circular blocks","block_length":40,"replicates":1999,
                     "seed":20261019,"interpretation":"nominal dependent-data interval within fixed markets"},
        "comparisons":comparisons,"new_model_innovation_claim":False,
        "original_paper_full_numerical_reproduction":False}
    write_json(public_dir/"SUMMARY.json",summary)
    return summary


def start_stage(stage,local_dir,public_dir,expected):
    current = identity()
    if current["fingerprint"] != expected:
        raise ValueError("Explicit pre-fit source identity does not match; no fitting performed")
    local_dir.mkdir(parents=True,exist_ok=True); public_dir.mkdir(parents=True,exist_ok=True)
    path = local_dir/f"START_{stage.upper()}.json"
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt["source_fingerprint"] != expected:
            raise ValueError("The existing stage belongs to a different source version")
        return
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    receipt = {"stage":stage,"utc":datetime.now(timezone.utc).isoformat(),"git_head_at_start":head,
               "source_fingerprint":expected,"source_files":current["files"],
               "no_2024_or_later_market_values":True,"planned_neural_fits":198,
               "planned_static_gas_fits":22,"normalizers":list(NORMALIZERS)}
    write_json(path,receipt); write_json(public_dir/path.name,receipt)


def main():
    parser = argparse.ArgumentParser()
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--print-source-hash",action="store_true")
    stage.add_argument("--prepare-only",action="store_true")
    stage.add_argument("--train-only",action="store_true")
    stage.add_argument("--aggregate-only",action="store_true")
    parser.add_argument("--base-prepared-dir",type=Path,default=Path("/tmp/carbon-conditional-persistence-runs/prepared"))
    parser.add_argument("--local-dir",type=Path,default=Path("/tmp/carbon-gas-normalization-run"))
    parser.add_argument("--public-dir",type=Path,default=ROOT/"results"/"gas_benchmark")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--expected-source-hash")
    args = parser.parse_args()
    if args.print_source_hash:
        print(json.dumps(identity(),indent=2)); return
    if not args.expected_source_hash:
        parser.error("Operational stages require --expected-source-hash frozen before fitting")
    local_dir,public_dir = private_root(args.local_dir),args.public_dir.resolve()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("Use one to four fixed CPU workers")
    name = "prepare" if args.prepare_only else ("train" if args.train_only else "aggregate")
    start_stage(name,local_dir,public_dir,args.expected_source_hash)
    if args.prepare_only:
        jobs = [(args.base_prepared_dir/f"{panel}.npz",h,local_dir,args.expected_source_hash)
                for panel in PANELS for h in HORIZONS]
        receipts = run_jobs(prepare_one,jobs,args.workers)
        write_json(public_dir/"GAS_STATIC_FIT_RECEIPTS.json",receipts)
    elif args.train_only:
        jobs = [(local_dir/"prepared"/f"{panel}_h{h}.npz",normalizer,seed,local_dir,args.expected_source_hash)
                for panel in PANELS for h in HORIZONS for normalizer in NORMALIZERS for seed in SEEDS]
        receipts = run_jobs(train_one,jobs,args.workers)
        write_json(public_dir/"NEURAL_FIT_RECEIPTS.json",receipts)
    else:
        aggregate(local_dir,public_dir,args.expected_source_hash)
    if identity()["fingerprint"] != args.expected_source_hash:
        raise ValueError("Source files changed during the stage; result provenance is invalid")
    print(json.dumps({"stage":name,"complete":True,"source_fingerprint":args.expected_source_hash}),flush=True)


if __name__ == "__main__":
    main()
