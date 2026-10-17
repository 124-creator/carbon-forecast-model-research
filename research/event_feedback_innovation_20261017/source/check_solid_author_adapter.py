"""Synthetic-only verification of actual SOLID author routine integration."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from solid_author_adapter import (SolidAuthorAdapter, array_hash,
                                  model_arrays, run_solid_author, verify_author)


class SyntheticData:
    def __init__(self, price, channels=2, start=180):
        self.price = np.asarray(price, dtype=float).copy()
        self.channels = channels
        self.start = start
        self.requests = []

    def at(self, t):
        if t < 60 or t >= len(self.price):
            raise ValueError("Synthetic input boundary")
        s = max(float(np.sqrt(np.mean(np.diff(self.price[t-60:t+1])**2))), .01)
        dp = np.diff(self.price[t-20:t+1])/s
        gap = np.ones(20)*np.log(2.)/np.log(8.)
        parts = [dp,gap]
        for j in range(self.channels-2):
            parts.append(np.sin(np.arange(t-19,t+1)/(j+3.)))
        return np.stack(parts,1),s

    def batch(self, origins, available_through, labels=True):
        ids=np.asarray(origins,dtype=int)
        if np.any(ids>available_through):
            raise ValueError("Synthetic unknown input")
        scales=np.array([self.at(int(s))[1] for s in ids])
        result=dict(origin=ids.copy(),scale=scales,base=self.price[ids].copy())
        self.requests.append(dict(origins=ids.tolist(),through=int(available_through),
                                  labels=bool(labels)))
        if labels:
            if np.any(ids+20>available_through):
                raise ValueError("Synthetic immature label")
            future=self.price[ids[:,None]+np.arange(1,21)]
            result["y"]=(future-result["base"][:,None])/scales[:,None]/np.sqrt(np.arange(1,21))
        return result


class SyntheticTeacher(nn.Module):
    def __init__(self, channels=2, cell="gru"):
        super().__init__()
        self.encoder={"gru":nn.GRU,"lstm":nn.LSTM}[cell](channels,32,batch_first=True)
        self.head=nn.Linear(32,20)

    def forward(self,x):
        z,_=self.encoder(x)
        return self.head(z[:,-1])


def mathematical_update(data, teacher, query, context_samples, neighbors, lr,
                        remove_distance):
    n=min(context_samples,query-20-60+1)
    candidates=np.arange(query-20-n+1,query-20+1)
    qx=data.at(query)[0].reshape(-1)
    if remove_distance:
        selected=candidates[::-1][:neighbors]
    else:
        # Exact author F.pairwise_distance default epsilon and stable sorting.
        distances=[float(np.linalg.norm(qx-data.at(int(s))[0].reshape(-1)+1e-6))
                   for s in candidates]
        selected=candidates[np.argsort(distances,kind="stable")[:neighbors]]
    weight=teacher.head.weight.detach().numpy().copy()
    bias=teacher.head.bias.detach().numpy().copy()
    gw=np.zeros_like(weight);gb=np.zeros_like(bias)
    with torch.no_grad():
        for s in selected:
            x=torch.tensor(data.at(int(s))[0][None])
            z=teacher.encoder(x)[0][0,-1].numpy()
            y=data.batch([int(s)],query)["y"][0]
            e=weight@z+bias-y
            gw+=2/20*np.outer(e,z)
            gb+=2/20*e
        qw=weight-lr*gw; qb=bias-lr*gb
        qz=teacher.encoder(torch.tensor(data.at(query)[0][None]))[0][0,-1].numpy()
    norm=qw@qz+qb
    raw=data.price[query]+data.at(query)[1]*np.sqrt(np.arange(1,21))*norm
    return dict(selected=selected,gw=gw,gb=gb,weight=qw,bias=qb,raw=raw)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/
                        "backup_review/SOLID_ADAPTER_SYNTHETIC.json")
    parser.add_argument("--author-root",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(20261017)
    price=100+np.cumsum(rng.normal(size=900))
    teacher=SyntheticTeacher()
    original_hash=array_hash(model_arrays(teacher))
    checks=[]
    max_errors={}
    started=time.perf_counter()
    for remove_distance in (False,True):
        base_data=SyntheticData(price)
        cached_artifacts=[]; uncached_artifacts=[]
        common=dict(author_root=args.author_root,context_samples=64,neighbors=10,
                    learning_rate=.01,remove_distance=remove_distance)
        queries=np.arange(650,654)
        cached,rc=run_solid_author(base_data,teacher,queries,cache_latents=True,
                                  artifact_sink=cached_artifacts.append,**common)
        uncached,ru=run_solid_author(SyntheticData(price),teacher,queries,
                                    cache_latents=False,artifact_sink=uncached_artifacts.append,
                                    **common)
        mode="recency" if remove_distance else "distance"
        err=float(np.max(np.abs(cached-uncached)))
        gerr=max(float(np.max(np.abs(a["summed_weight_gradient"]-
                                     b["summed_weight_gradient"])))
                 for a,b in zip(cached_artifacts,uncached_artifacts))
        assert err<1e-12 and gerr<1e-12
        max_errors[mode+"_cached_uncached_prediction"]=err
        max_errors[mode+"_cached_uncached_gradient"]=gerr
        checks.append(mode+"_cache_prediction_and_gradient_identity")
        for a in cached_artifacts:
            expected=mathematical_update(SyntheticData(price),teacher,int(a["origin"]),64,10,.01,
                                         remove_distance)
            assert np.array_equal(a["selected_origins"],expected["selected"])
            err=max(float(np.max(np.abs(a["summed_weight_gradient"]-expected["gw"]))),
                    float(np.max(np.abs(a["summed_bias_gradient"]-expected["gb"]))),
                    float(np.max(np.abs(a["adapted_head_weight"]-expected["weight"]))),
                    float(np.max(np.abs(a["raw_prediction"]-expected["raw"]))))
            assert err<1e-11
            max_errors[mode+"_mathematical_update"]=max(
                max_errors.get(mode+"_mathematical_update",0.),err)
        checks.append(mode+"_author_sum_step_matches_independent_formula")
        assert rc["head_updates"]==4 and rc["neighbor_backward_calls"]==40
        assert all(r["maximum_label_index"]<=r["origin"] and r["actual_selected_count"]==10
                   for r in rc["diagnostics"])
        assert all(not r["query_future_label_supplied"] for r in rc["diagnostics"])
        assert all(max(x["origins"])+20<=x["through"] for x in base_data.requests if x["labels"])
        checks.append(mode+"_actual_steps_and_maturity")
        isolated,_=run_solid_author(SyntheticData(price),teacher,[653],cache_latents=True,**common)
        assert np.max(np.abs(isolated[0]-cached[-1]))<1e-12
        checks.append(mode+"_query_reset")
        changed=price.copy();changed[651:]+=10000+rng.normal(size=len(price)-651)
        suffix_artifacts=[]
        suffix,_=run_solid_author(SyntheticData(changed),teacher,[650],cache_latents=True,
                                 artifact_sink=suffix_artifacts.append,**common)
        assert np.array_equal(suffix[0],cached[0])
        assert np.array_equal(suffix_artifacts[0]["summed_weight_gradient"],
                              cached_artifacts[0]["summed_weight_gradient"])
        checks.append(mode+"_future_suffix_prediction_and_gradient_identity")
        if remove_distance:
            assert rc["encoder_calls"]<ru["encoder_calls"]
            checks.append("cache_reduces_actual_encoder_evaluations")
    # Early support clips the pool, with no negative index or future filler.
    clipped,rr=run_solid_author(SyntheticData(price),teacher,[180],
                               author_root=args.author_root,context_samples=512,neighbors=10,
                               learning_rate=.001)
    assert rr["diagnostics"][0]["candidate_count"]==101
    assert rr["diagnostics"][0]["candidate_first"]==60
    checks.append("candidate_pool_clips_to_available_complete_H20")
    for channels,cell in [(5,"gru"),(2,"lstm")]:
        t=SyntheticTeacher(channels,cell)
        out,_=run_solid_author(SyntheticData(price,channels),t,[650],
                               author_root=args.author_root,context_samples=32,neighbors=5)
        assert out.shape==(1,20) and np.isfinite(out).all()
    checks.append("GRU_C5_and_LSTM_C2_shape")
    try:
        run_solid_author(SyntheticData(price),teacher,[650],author_root=args.author_root,
                         deadline=time.monotonic()-1)
    except TimeoutError:
        checks.append("expired_deadline_stops_before_query_update")
    else:
        raise AssertionError("Expired deadline accepted")
    assert array_hash(model_arrays(teacher))==original_hash
    checks.append("caller_offline_checkpoint_unchanged")
    source=Path(__file__).with_name("solid_author_adapter.py")
    receipt=dict(status="PASS",market_data_reads=0,market_fits=0,
                 synthetic_only=True,checks=checks,max_abs_errors=max_errors,
                 author=verify_author(args.author_root),
                 adapter_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 runtime=dict(torch=torch.__version__,numpy=np.__version__,
                              threads=torch.get_num_threads(),dtype="float64"),
                 seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
