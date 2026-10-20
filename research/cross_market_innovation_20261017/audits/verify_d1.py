"""Independent D1 replay with NumPy only; NEVER trains or reads confirmation.

Reconstructs chronological supports, all original labels and input views,
GRU/LSTM/ridge forward predictions, validation scores, per-cell scores, selected
epoch logic, parameter hashes and completed fit accounting.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from statistical_reference import reference

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "carbon_workspace/research/cross_market_innovation_20261017"
OUT = HERE / "d1_audit"
H = np.arange(1,21)
VERTICES = {"full":np.arange(21), "short":np.arange(10,21), "sparse":np.arange(0,21,2)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(state):
    value = hashlib.sha256()
    for key, array in sorted(state.items()):
        value.update(key.encode())
        value.update(str(array.shape).encode())
        value.update(array.tobytes())
    return value.hexdigest()


class Comparisons:
    def __init__(self):
        self.elements = 0
        self.maximum_error = 0.
        self.largest = None

    def equal(self, a, b, name):
        aa, bb = np.asarray(a), np.asarray(b)
        assert aa.shape == bb.shape, (name, aa.shape, bb.shape)
        assert np.array_equal(aa, bb), name
        self.elements += aa.size

    def close(self, a, b, name, atol=2e-9, rtol=2e-12):
        aa, bb = np.asarray(a), np.asarray(b)
        assert aa.shape == bb.shape, (name, aa.shape, bb.shape)
        assert np.isfinite(aa).all() and np.isfinite(bb).all(), name
        error = float(np.max(np.abs(aa-bb))) if aa.size else 0.
        if error > self.maximum_error:
            self.maximum_error, self.largest = error, name
        assert np.allclose(aa, bb, atol=atol, rtol=rtol), (name, error)
        self.elements += aa.size


def load_sources(plan):
    registry_path = ROOT / "provenance/DEVELOPMENT_REGISTRY.json"
    assert digest(registry_path) == plan["registry_sha256"]
    registry = json.loads(registry_path.read_text())
    markets = {}
    metadata = []
    for record in registry:
        path = Path(record["path"])
        assert "confirmation" not in str(path).lower()
        assert "/local_only/data/development/" in str(path)
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        dates = np.array([row["date"][:10] for row in rows])
        # These files must already be the development partition. No suffix read
        # from the original complete Shanghai source or sealed confirmation.
        assert len(dates) and np.all(dates < "2024-01-01") and np.all(dates[1:] > dates[:-1])
        price = np.array([float(row[record["price_column"]]) for row in rows])
        assert np.isfinite(price).all() and np.all(price > 0)
        days = np.array([dt.date.fromisoformat(d).toordinal() for d in dates])
        markets[record["name"]] = dict(price=price, dates=dates, days=days)
        spans = days[20:]-days[:-20]
        metadata.append(dict(name=record["name"], rows=len(price), sha256=record["sha256"],
                             first_date=str(dates[0]), last_date=str(dates[-1]),
                             observations_by_year={y:int(np.sum(np.char.startswith(dates,y)))
                                                   for y in sorted(set(d[:4] for d in dates))},
                             h20_median_calendar_days=float(np.median(spans)),
                             h20_maximum_calendar_days=int(spans.max())))
    return markets, metadata


def support(market, cutoff):
    end = int(np.searchsorted(market["dates"], cutoff, side="left"))
    # Literal index arithmetic, without importing the production support helper.
    last_mature = end-21
    valid = np.arange(last_mature-79, last_mature+1)
    train = np.arange(20, valid[0]-20)
    assert len(train) >= 120 and train[-1]+20 < valid[0] and valid[-1]+20 < end
    first, stop = int(train[0]-20), int(train[-1]+1)
    increments = np.diff(market["price"][first:stop])
    scale = max(float(np.sqrt(np.mean(increments**2))), 1e-8)
    return dict(end=end, train=train, validation=valid, scale=scale,
                scale_start=first, scale_stop=stop, cutoff=cutoff, cap=None)


def pack(markets, names, cutoff, role):
    chunks, supports = [], {}
    for mid, name in enumerate(names):
        market = markets[name]
        sup = support(market, cutoff)
        supports[name] = sup
        if role == "evaluation":
            year = cutoff[:4]
            origins = np.array([t for t in range(20,len(market["price"])-20)
                                if market["dates"][t][:4] == year
                                and market["dates"][t+20][:4] == year], dtype=np.int64)
        else:
            origins = sup[role]
        p, dates, days, scale = market["price"], market["dates"], market["days"], sup["scale"]
        raw_y = p[origins[:,None]+H] - p[origins,None]
        # The causal original reference may be evaluated once through the last
        # requested origin; it never receives the forecast target suffix here.
        anchors = reference(p[:int(origins.max())+1])[0][origins]
        row = dict(origin=origins, dates=dates[origins], market=np.full(len(origins),mid),
                   scale=np.full(len(origins),scale), raw_y=raw_y, stat_raw=anchors,
                   y=raw_y / scale / np.sqrt(H), stat_normalized=anchors / scale / np.sqrt(H))
        for view, indices in VERTICES.items():
            pids = origins[:,None]-20+indices
            px, dx = p[pids], days[pids]
            n = len(indices)-1
            x = np.zeros((len(origins),20,3))
            x[:,:n,0] = (px[:,1:]-px[:,:-1]) / scale
            x[:,:n,1] = np.log1p(np.diff(indices))
            x[:,:n,2] = np.log1p(dx[:,1:]-dx[:,:-1])
            row[view] = x
            row[view+"_mask"] = np.broadcast_to(np.arange(20)<n, (len(origins),20)).copy()
        chunks.append(row)
    data = {key:np.concatenate([row[key] for row in chunks]) for key in chunks[0]}
    data["market_names"] = list(names)
    return data, supports


def sigmoid(x):
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1/(1+np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e/(1+e)
    return out


def predict_normalized(state, config, data, view):
    family = config["family"]
    ids = np.eye(len(data["market_names"]))[data["market"]]
    if family == "ridge":
        return np.c_[data[view].reshape(len(ids),-1),ids] @ state["beta"]
    price_only = family in ("gru_price", "lstm_price", "gru_stat_price")
    x = data[view][:,:,:1] if price_only else data[view]
    mask = data[view+"_mask"]
    repeated_id = np.broadcast_to(ids[:,None], (len(ids),20,ids.shape[1])) * mask[:,:,None]
    x = np.concatenate([x,repeated_id],axis=2)
    hidden = state["head.weight"].shape[1]
    h = np.zeros((len(ids),hidden))
    c = np.zeros_like(h)
    wi, wh = state["gru.weight_ih_l0"], state["gru.weight_hh_l0"]
    bi, bh = state["gru.bias_ih_l0"], state["gru.bias_hh_l0"]
    for step in range(20):
        input_gates = x[:,step] @ wi.T + bi
        hidden_gates = h @ wh.T + bh
        if family.startswith("lstm"):
            i,f,g,o = np.split(input_gates+hidden_gates,4,axis=1)
            new_c = sigmoid(f)*c + sigmoid(i)*np.tanh(g)
            new_h = sigmoid(o)*np.tanh(new_c)
            c = np.where(mask[:,step,None],new_c,c)
        else:
            ir, iz, inn = np.split(input_gates,3,axis=1)
            hr, hz, hn = np.split(hidden_gates,3,axis=1)
            reset, update = sigmoid(ir+hr), sigmoid(iz+hz)
            candidate = np.tanh(inn+reset*hn)
            new_h = (1-update)*candidate+update*h
        h = np.where(mask[:,step,None],new_h,h)
    value = h @ state["head.weight"].T + state["head.bias"]
    if family == "gru_stat_price":
        assert view == "full"
        value = value + data["stat_normalized"]
    return value


def cell_metrics(data, forecast, view):
    cells = []
    years = np.array([date[:4] for date in data["dates"]])
    for mid, name in enumerate(data["market_names"]):
        for year in sorted(set(years[data["market"]==mid])):
            use = (data["market"]==mid) & (years==year)
            for h in (1,5,20):
                y, p = data["raw_y"][use,h-1], forecast[use,h-1]
                mse, rw = float(np.mean((p-y)**2)), float(np.mean(y**2))
                cells.append(dict(market=name,year=int(year),horizon=h,origins=int(use.sum()),
                                  mse=mse,rw_mse=rw,relative_mse=mse/rw,
                                  mae=float(np.mean(abs(p-y))),view=view))
    return dict(score=float(np.mean([c["relative_mse"] for c in cells])), cells=cells)


def validate_score(data, normalized):
    ratios = []
    for mid in np.unique(data["market"]):
        use = data["market"] == mid
        y, p = data["y"][use][:,[0,4,19]], normalized[use][:,[0,4,19]]
        ratios.append(np.mean(np.mean((y-p)**2,axis=0)/np.mean(y**2,axis=0)))
    return float(np.mean(ratios))


def metric_check(cmp, computed, saved, prefix):
    cmp.close(computed["score"],saved["score"],prefix+"/score")
    assert len(computed["cells"]) == len(saved["cells"])
    for index,(a,b) in enumerate(zip(computed["cells"],saved["cells"])):
        for key in ("market","year","horizon","origins","view"):
            assert a[key] == b[key], (prefix,index,key)
        for key in ("mse","rw_mse","relative_mse","mae"):
            cmp.close(a[key],b[key],f"{prefix}/cell{index}/{key}")


def audit(partial=False):
    OUT.mkdir(exist_ok=True)
    plan_path = ROOT / "D1_PRE_RUN.json"
    plan = json.loads(plan_path.read_text())
    started = json.loads((ROOT/"D1_STARTED.json").read_text())
    assert started["plan_sha256"] == digest(plan_path)
    assert plan["formal_predictor_fits"] == len(plan["tasks"]) == 141
    for name, sha in plan["source_sha256"].items():
        assert digest(ROOT/"frozen_d1_source"/name) == sha, name
    assert not plan["confirmation_files_allowed"] and plan["development_end_exclusive"] == "2024-01-01"
    markets, metadata = load_sources(plan)
    cmp = Comparisons()
    caches, audited, initial_groups = {}, [], {}
    for task in plan["tasks"]:
        tid = task["id"]
        rp = ROOT/"d1_receipts"/(tid+".json")
        sp = ROOT/"local_only/d1"/(tid+"_state.npz")
        pp = ROOT/"local_only/d1"/(tid+"_predictions.npz")
        if not (rp.exists() and sp.exists() and pp.exists()):
            if partial:
                continue
            raise FileNotFoundError(tid+" incomplete")
        receipt = json.loads(rp.read_text())
        assert receipt["task"] == task and receipt["config"] == task["config"]
        assert receipt["seed"] == task["seed"] == 17
        assert receipt["fitting_code_sources"] == plan["source_sha256"]
        assert not receipt["confirmation_data_used"]
        key = (tuple(task["market_names"]),task["cutoff"])
        if key not in caches:
            train, support_rows = pack(markets,*key,"train")
            valid, _ = pack(markets,*key,"validation")
            evaluate, _ = pack(markets,*key,"evaluation")
            caches[key] = (train,valid,evaluate,support_rows)
        train,valid,evaluate,support_rows = caches[key]
        for name, expected in support_rows.items():
            observed = receipt["supports"][name]
            for field in ("train","validation"):
                cmp.equal(expected[field],observed[field],tid+"/support/"+name+"/"+field)
            for field in ("end","scale_start","scale_stop","cutoff","cap"):
                assert observed[field] == expected[field], (tid,name,field)
            cmp.close(expected["scale"],observed["scale"],tid+"/scale/"+name,atol=1e-14)
        with np.load(sp,allow_pickle=False) as stream:
            state = {k:stream[k] for k in stream.files}
        assert state_hash(state) == receipt["state_sha256"]
        assert sum(x.size for x in state.values()) == receipt["parameter_count"]
        with np.load(pp,allow_pickle=False) as stream:
            saved = {k:stream[k] for k in stream.files}
        cmp.equal(saved["market_names"],evaluate["market_names"],tid+"/market_order")
        for field in ("origin","dates","market"):
            cmp.equal(saved[field],evaluate[field],tid+"/"+field)
        for field in ("raw_y","scale","stat_raw"):
            cmp.close(saved[field],evaluate[field],tid+"/"+field,atol=1e-13)
        views = ["full"] if task["config"]["family"] == "gru_stat_price" else list(VERTICES)
        all_metrics = {}
        for view in views:
            forecast = predict_normalized(state,task["config"],evaluate,view)
            forecast = forecast * evaluate["scale"][:,None] * np.sqrt(H)
            cmp.close(forecast,saved["prediction_"+view],tid+"/numpy_forward/"+view)
            value = cell_metrics(evaluate,forecast,view)
            metric_check(cmp,value,receipt["development_metrics"][view],tid+"/"+view)
            all_metrics[view] = value
        val_pred = predict_normalized(state,task["config"],valid,"full")
        vscore = validate_score(valid,val_pred)
        cmp.close(vscore,receipt["validation_score"],tid+"/selected_validation")
        epochs, selected = receipt["epochs"],receipt["selected_epoch"]
        assert epochs == len(receipt["trace"]) <= plan["maximum_epochs"]
        assert 0 <= selected <= epochs
        if task["config"]["family"] != "ridge":
            best, found, stale = receipt["initial_validation"],0,0
            for epoch,row in enumerate(receipt["trace"],1):
                assert row["epoch"] == epoch
                augmented = task["config"]["family"] in ("augment","kcm","pointwise","kcm_unblocked")
                expected_view = ("short" if epoch%2 else "sparse") if augmented else "full"
                assert row["view"] == expected_view
                assert np.isfinite([v for v in row.values() if isinstance(v,(int,float))]).all()
                assert row["mechanism_penalty"] >= 0
                if row["validation_score"] < best-1e-6:
                    best,found,stale = row["validation_score"],epoch,0
                else:
                    stale += 1
                assert stale < 12 or epoch == epochs
            assert found == selected
            cmp.close(best,receipt["validation_score"],tid+"/epoch_selection")
            assert epochs == 60 or stale == 12
            if task["config"]["family"] in ("gru","augment","kcm"):
                initial_key = key+(task["config"]["hidden"],task["config"]["shrink"])
                prior = initial_groups.setdefault(initial_key,receipt["initial_state_sha256"])
                assert receipt["initial_state_sha256"] == prior
            if task["config"]["family"] == "kcm":
                for rec in receipt["kernel_audit"]:
                    names = task["market_names"]
                    origins = support_rows[names[rec["market"]]]["train"]
                    assert np.all(np.diff(origins) == 1)
                    # For a contiguous run of n origins, sum the two directed
                    # pairs for lags 41,...,n-1 analytically.
                    pair_count = max(len(origins)-41,0) * max(len(origins)-40,0)
                    assert rec["ordered_pairs"] == pair_count and rec["minimum_origin_separation"] == 41
                    assert rec["fixed_feature_dimension"] == 60
        audited.append(dict(id=tid,task=task,metrics=all_metrics,selected_epoch=selected,
                            epochs=epochs,validation_score=vscore,receipt_sha256=digest(rp),
                            state_file_sha256=digest(sp),predictions_file_sha256=digest(pp)))
    attempts = [json.loads(line) for line in (ROOT/"D1_ATTEMPTS.jsonl").read_text().splitlines()]
    ids = [r["task_id"] for r in attempts]
    assert len(ids) == len(set(ids)) and ids == [t["id"] for t in plan["tasks"][:len(ids)]]
    completed_path = ROOT/"D1_RUN_COMPLETE.json"
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else None
    if not partial:
        assert completed and completed["all_completed"] and completed["completed_fits"] == 141
        assert completed["additional_retry_fits"] == 0 and not completed["confirmation_used"]
        assert len(audited) == len(attempts) == 141
    result = dict(status="PARTIAL_PASS" if partial else "PASS", audit_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  scope="D1 development only; independent NumPy replay, no training",expected_fits=141,
                  audited_fits=len(audited),observed_attempts=len(attempts),completed_receipt=completed,
                  market_fits_by_auditor=0,optimizer_steps_by_auditor=0,confirmation_file_reads=0,
                  independent_checks=cmp.elements,maximum_absolute_difference=cmp.maximum_error,
                  largest_difference_location=cmp.largest,plan_sha256=digest(plan_path),
                  source_sha256=plan["source_sha256"],numpy_version=np.__version__,
                  audit_script_sha256=digest(Path(__file__)),development_sources=metadata,
                  checks=["frozen code and registry identities", "fully mature chronological train/validation",
                          "training-only RMS scale", "same-year H20 evaluation support",
                          "all original labels and statistical anchors", "market ID order and raw/coarse inputs",
                          "independent GRU/LSTM/ridge NumPy forward", "all view/horizon point predictions",
                          "all reported cell scores and selected-state validation", "epoch selection logic",
                          "matched neural initial-state hashes", "no duplicate fit attempts"],
                  limitations=["Initial random-encoder/ridge parameters were hashed but not saved by D1; their initial values are not independently reconstructed.",
                               "Only the saved selected state's validation prediction is replayed; intermediate epoch parameters were not stored.",
                               "This verifies implementation and scores, not novelty or statistical significance."])
    label = f"PARTIAL_{len(audited):03d}" if partial else "COMPLETE"
    (OUT/f"AUDIT_D1_{label}.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    (OUT/f"RECOMPUTED_D1_{label}.json").write_text(json.dumps(audited,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","audited_fits","independent_checks",
                                         "maximum_absolute_difference","largest_difference_location")},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial",action="store_true")
    audit(parser.parse_args().partial)
