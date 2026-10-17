"""Frozen DSOF author-core execution anchor; development data only.

Own experiment runner. No author code or auction observations are redistributed.
Modes: --plan (no fit), --pilot (one timing run), --formal (one scored run).
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from author_adapter import make_author_dsof


STUDY = Path(__file__).resolve().parents[1]
PROTOCOL = STUDY / "provenance/AUTHOR_EXECUTION_PROTOCOL.json"
PROTOCOL_SHA256 = "3cc6170b39d793612abd3e07b320c5b94874c5055023dfc90bd65e930ec5bc16"
PRIVATE = STUDY / "local_only/author_execution"
PUBLIC = STUDY / "provenance"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def hash_state(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        arr = state[name].detach().cpu().numpy()
        digest.update(name.encode() + b"\0" + str(arr.dtype).encode() + b"\0")
        digest.update(str(arr.shape).encode() + b"\0" + arr.tobytes())
    return digest.hexdigest()


def snapshot(state):
    return {k: v.detach().clone() for k, v in state.items()}


def deadline_check(deadline):
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("Frozen execution deadline reached; no retry")


def load_inputs(protocol):
    path = Path(protocol["data_path"])
    if sha(path) != protocol["data_sha256"]:
        raise ValueError("Approved development file hash mismatch")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    dates = np.array([row["date"] for row in rows])
    price = np.array([float(row["price"]) for row in rows], dtype=np.float64)
    if not np.isfinite(price).all() or not np.all(dates[1:] > dates[:-1]):
        raise ValueError("Nonfinite price or unsorted/duplicate dates")
    if np.any(dates >= "2024-01-01"):
        raise ValueError("Confirmation/future rows are forbidden in this runner")
    L, H = protocol["context"], protocol["horizon"]
    cutoff = int(np.searchsorted(dates, protocol["offline_cutoff"]))
    eligible = np.arange(L-1, cutoff-H, dtype=np.int64)
    if len(eligible) < 120:
        raise ValueError("Insufficient pre-cutoff support")
    val = eligible[-80:]
    train = np.arange(L-1, int(val[0])-H, dtype=np.int64)
    if len(train) < 64 or train[-1]+H >= val[0] or val[-1]+H >= cutoff:
        raise ValueError("Training/validation temporal purge violation")
    scale_end = int(val[0])
    mean = float(price[:scale_end].mean())
    sd = float(price[:scale_end].std(ddof=0))
    if sd <= 0:
        raise ValueError("Zero training price standard deviation")
    norm = (price-mean)/sd
    online = np.arange(cutoff, len(price), dtype=np.int64)
    score = np.array([t for t in online if t+H < len(price)
                      and int(dates[t][:4]) in protocol["formal"]["scored_years"]
                      and dates[t][:4] == dates[t+H][:4]], dtype=np.int64)
    return dict(dates=dates, price=price, norm=norm, train=train, val=val,
                online=online, score=score, mean=mean, sd=sd, scale_end=scale_end,
                cutoff=cutoff, context=L, horizon=H)


def tensor_windows(data, origins):
    L, H = data["context"], data["horizon"]
    x = data["norm"][origins[:, None] + np.arange(-L+1, 1)]
    y = data["norm"][origins[:, None] + np.arange(1, H+1)]
    return torch.tensor(x[:, :, None]), torch.tensor(y[:, :, None])


def native_predict(net, data, origins):
    x, _ = tensor_windows(data, origins)
    pred = net.predict(x)[:, :, 0].numpy()*data["sd"] + data["mean"]
    if not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite model predictions")
    return pred


def relative_score(price, origins, pred, horizons=(1, 5, 20)):
    cells = []
    for h in horizons:
        y = price[origins+h]
        mse = float(np.mean((pred[:, h-1]-y)**2))
        rw = float(np.mean((price[origins]-y)**2))
        if rw <= 0 or not np.isfinite(mse):
            raise FloatingPointError("Invalid score denominator or loss")
        cells.append(dict(horizon=h, n=len(origins), model_mse=mse, rw_mse=rw,
                          relative_mse=mse/rw))
    return float(np.mean([c["relative_mse"] for c in cells])), cells


def support_metadata(data):
    out = {}
    for key in ("train", "val", "online", "score"):
        origins = data[key]
        out[key] = {"count": len(origins), "origin_min": int(origins[0]),
                    "origin_max": int(origins[-1]),
                    "first_date": str(data["dates"][origins[0]]),
                    "last_date": str(data["dates"][origins[-1]]),
                    "origin_array_sha256": hashlib.sha256(origins.tobytes()).hexdigest()}
    out["scale_price_end_exclusive"] = data["scale_end"]
    out["scale_last_date"] = str(data["dates"][data["scale_end"]-1])
    out["train_max_label_index"] = int(data["train"][-1]+data["horizon"])
    out["val_min_origin_index"] = int(data["val"][0])
    out["val_max_label_index"] = int(data["val"][-1]+data["horizon"])
    out["offline_cut_index"] = data["cutoff"]
    return out


def execute(mode, author_root):
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    adapter_path = Path(__file__).with_name("author_adapter.py")
    if sha(adapter_path) != protocol["adapter_sha256"]:
        raise ValueError("Tested adapter changed")
    deadline = datetime.fromisoformat(protocol["deadline_utc"])
    deadline_check(deadline)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = load_inputs(protocol)
    source = {"runner_sha256":sha(__file__), "adapter_sha256":sha(adapter_path),
              "protocol_sha256":sha(PROTOCOL), "author_commit":protocol["author_commit"]}
    plan = {"created_utc":now(), "source":source, "support":support_metadata(data),
            "data_sha256":protocol["data_sha256"], "confirmation_access":False,
            "runtime":{"python":platform.python_version(), "numpy":np.__version__,
                       "torch":torch.__version__, "torch_threads":torch.get_num_threads()},
            "selection_min_delta":0.0, "early_stopping":"strict decrease; epoch zero eligible"}
    if mode == "plan":
        dump(PUBLIC / "AUTHOR_ANCHOR_EXECUTION_PLAN.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    marker = PRIVATE / f"{mode.upper()}_STARTED.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    np.savez_compressed(PRIVATE / f"{mode.upper()}_SUPPORTS.npz",
                        training_origins=data["train"], validation_origins=data["val"],
                        online_origins=data["online"], scored_origins=data["score"],
                        dates=data["dates"], scaler_mean=data["mean"], scaler_sd=data["sd"])
    start = time.perf_counter()
    status = "INCOMPLETE"
    result = dict(plan, phase=mode, status=status)
    try:
        seed = protocol["pilot"]["seed"] if mode == "pilot" else protocol["formal"]["seed"]
        net = make_author_dsof(author_root, context=data["context"], horizon=data["horizon"],
                               seed=seed, replay_size=protocol["replay_size"],
                               replay_batch=protocol["replay_batch"], gamma=protocol["gamma"],
                               td_k=tuple(protocol["td_k"]), dtype=torch.float64)
        rng = np.random.default_rng(seed)
        xtrain, ytrain = tensor_windows(data, data["train"])
        initial_state_hash = hash_state(net.state_dict())
        n_epochs = protocol["pilot"]["offline_epochs"] if mode == "pilot" else protocol["formal"]["max_epochs"]
        batch_size = protocol["formal"]["offline_batch_size"]
        trace, stale, best_epoch = [], 0, 0
        best_state = snapshot(net.state_dict())
        if mode == "formal":
            initial_val_predictions = native_predict(net, data, data["val"])
            best_score, _ = relative_score(data["price"], data["val"], initial_val_predictions)
            trace.append({"epoch":0, "validation_score":best_score, "state_hash":initial_state_hash})
        else:
            best_score = None
        epoch_seconds = []
        offline_start = time.perf_counter()
        for epoch in range(1, n_epochs+1):
            deadline_check(deadline)
            epoch_start = time.perf_counter()
            order = rng.permutation(len(xtrain))
            loss_sum = 0.0
            for first in range(0, len(order), batch_size):
                deadline_check(deadline)
                idx = order[first:first+batch_size]
                loss = net.pretrain_batch(xtrain[idx], ytrain[idx])["loss"]
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite author offline loss; no retry")
                loss_sum += float(loss)*len(idx)
            sec = time.perf_counter()-epoch_start
            epoch_seconds.append(sec)
            if mode == "formal":
                vp = native_predict(net, data, data["val"])
                score, _ = relative_score(data["price"], data["val"], vp)
                improved = score < best_score
                if improved:
                    best_score, best_epoch, best_state, stale = score, epoch, snapshot(net.state_dict()), 0
                else:
                    stale += 1
                trace.append(dict(epoch=epoch, training_teacher_mse=loss_sum/len(order),
                                  validation_score=score, seconds=sec, selected=improved,
                                  state_hash=hash_state(net.state_dict())))
                if stale >= protocol["formal"]["patience"]:
                    break
            else:
                # Timing pilot deliberately does not compute validation/scored losses.
                trace.append(dict(epoch=epoch, seconds=sec, batches=int(np.ceil(len(order)/batch_size)), finite=True))
        offline_seconds = time.perf_counter()-offline_start
        if mode == "formal":
            net.load_state_dict(best_state)
        offline_hash = hash_state(net.state_dict())
        offline_checkpoint = snapshot(net.state_dict())
        torch.save({"state":offline_checkpoint, "selected_epoch":best_epoch if mode == "formal" else None},
                   PRIVATE / f"{mode.upper()}_OFFLINE_STATE.pt")
        selected_val_predictions = native_predict(net, data, data["val"]) if mode == "formal" else None
        training_predictions = native_predict(net, data, data["train"]) if mode == "formal" else None
        online_origins = data["online"][:protocol["pilot"]["online_observations"]] if mode == "pilot" else data["online"]
        predictions = []
        online_trace = []
        online_start = time.perf_counter()
        for t in online_origins:
            deadline_check(deadline)
            event_start = time.perf_counter()
            pred = net.online_event(data["norm"][:int(t)+1], origin=int(t))[0, :, 0].numpy()
            if not np.isfinite(pred).all():
                raise FloatingPointError("Nonfinite author online prediction; no retry")
            predictions.append(pred*data["sd"]+data["mean"])
            online_trace.append(dict(origin=int(t), max_available_index=int(t),
                                     matured_origin=int(t)-data["horizon"],
                                     teacher_slow_steps=net.optimizers["teacher_slow"].nonempty_gradient_steps,
                                     student_fast_steps=net.optimizers["student_fast"].nonempty_gradient_steps,
                                     state_hash=hash_state(net.state_dict()),
                                     seconds=time.perf_counter()-event_start))
        online_seconds = time.perf_counter()-online_start
        predictions = np.array(predictions)
        result.update(status="COMPLETE", seed=seed, offline_epochs=len(trace)-(mode == "formal"),
                      selected_epoch=best_epoch if mode == "formal" else None,
                      seconds=time.perf_counter()-start, offline_seconds=offline_seconds,
                      online_seconds=online_seconds, mean_epoch_seconds=float(np.mean(epoch_seconds)),
                      online_observations=len(online_origins), initial_state_hash=initial_state_hash,
                      offline_state_hash=offline_hash, final_state_hash=hash_state(net.state_dict()),
                      author_execution=net.execution_receipt(), finished_utc=now())
        if mode == "pilot":
            steady = float(np.mean([r["seconds"] for r in online_trace[-20:]]))
            result["estimated_formal_upper_seconds"] = 2.0*(
                float(np.mean(epoch_seconds))*protocol["formal"]["max_epochs"]
                + max(steady, online_seconds/max(1,len(online_origins)))*len(data["online"]))
            result["estimate_rule"] = "2x (mean offline epoch x40 + max(last20 event mean, all-event mean) x all online events)"
            result["prediction_shape"] = list(predictions.shape)
            result["model_selection_or_forecast_score_computed"] = False
        else:
            selected = np.searchsorted(online_origins, data["score"])
            cells = []
            for year in protocol["formal"]["scored_years"]:
                keep = np.array([str(d).startswith(str(year)) for d in data["dates"][data["score"]]])
                _, part = relative_score(data["price"], data["score"][keep], predictions[selected[keep]])
                cells.extend([dict(year=year, **row) for row in part])
            result["cells"] = cells
            result["primary_score"] = float(np.mean([row["relative_mse"] for row in cells]))
            result["selected_validation_score"] = best_score
            np.savez_compressed(PRIVATE / "AUTHOR_ANCHOR_PREDICTIONS.npz",
                                origins=online_origins, predictions=predictions,
                                training_origins=data["train"], training_predictions=training_predictions,
                                validation_origins=data["val"], validation_predictions=selected_val_predictions,
                                initial_validation_predictions=initial_val_predictions,
                                scored_origins=data["score"], dates=data["dates"],
                                scaler_mean=data["mean"], scaler_sd=data["sd"],
                                **{"offline_"+k:v.numpy() for k,v in offline_checkpoint.items()},
                                **{"final_"+k:v.detach().numpy() for k,v in net.state_dict().items()})
            torch.save({"selected_offline_state":offline_checkpoint, "final_state":net.state_dict(),
                        "optimizer_states":{k:v.optimizer.state_dict() for k,v in net.optimizers.items()},
                        "replay_buffer":net.replayBuffer, "laststepA":net.laststepA,
                        "laststepB":net.laststepB, "numpy_rng_state":net.numpy_rng_state,
                        "forecast_records":net.forecast_records}, PRIVATE / "AUTHOR_ANCHOR_STATES.pt")
        dump(PRIVATE / f"{mode.upper()}_OFFLINE_TRACE.json", trace)
        dump(PRIVATE / f"{mode.upper()}_ONLINE_TRACE.json", online_trace)
    except Exception as exc:
        result.update(status="INCOMPLETE", error_type=type(exc).__name__, error=str(exc),
                      seconds=time.perf_counter()-start, finished_utc=now())
        if "trace" in locals():
            dump(PRIVATE / f"{mode.upper()}_OFFLINE_TRACE.json", trace)
        if "online_trace" in locals():
            dump(PRIVATE / f"{mode.upper()}_ONLINE_TRACE.json", online_trace)
        if "net" in locals():
            torch.save({"partial_state":net.state_dict()}, PRIVATE / f"{mode.upper()}_PARTIAL_STATE.pt")
        dump(PUBLIC / f"AUTHOR_ANCHOR_{mode.upper()}.json", result)
        raise
    dump(PUBLIC / f"AUTHOR_ANCHOR_{mode.upper()}.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true")
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--formal", action="store_true")
    parser.add_argument("--author-root", type=Path,
                        default=STUDY / "local_only/reference/author_dsof")
    args = parser.parse_args()
    execute("plan" if args.plan else "pilot" if args.pilot else "formal", args.author_root)
