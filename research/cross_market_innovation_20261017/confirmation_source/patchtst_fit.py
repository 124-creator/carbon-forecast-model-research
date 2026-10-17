"""Bounded PatchTST fitting for project D2; no import-time training or data reads.

Inputs are data_core pack dictionaries augmented by the CALLER with
`levels`: raw price levels [N,21] at origin-20,...,origin. `scale` is [N],
estimated from the corresponding training support by data_core. The helper never
opens a market file, looks up future prices, or modifies D1 source/protocol.

Pass `common=fit_core` to reuse its configure, group_weights, validation_score,
state_arrays and state_hash helpers. An omitted common imports fit_core lazily.
"""
from __future__ import annotations

import copy
import importlib
import time

import numpy as np
import torch

from patchtst_adapter import LocalPatchTST


def _common_helpers(common=None):
    common = importlib.import_module("fit_core") if common is None else common
    for name in ("configure", "group_weights", "validation_score", "state_arrays", "state_hash"):
        if not callable(getattr(common, name, None)):
            raise TypeError("common must provide callable " + name)
    return common


def _inputs(data, *, check_target: bool = True):
    levels = np.asarray(data["levels"], dtype=np.float64)
    scale = np.asarray(data["scale"], dtype=np.float64)
    market = np.asarray(data["market"], dtype=np.int64)
    if levels.ndim != 2 or levels.shape[1] != 21 or not len(levels):
        raise ValueError("Attach raw levels with shape [N,21], including current price.")
    n = len(levels)
    if scale.shape != (n,) or market.shape != (n,):
        raise ValueError("scale and market must each have shape [N], not [N,1].")
    if not np.isfinite(levels).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Nonfinite levels or nonpositive training scale.")
    # Catch accidental normalized-level inputs, wrong origin alignment or scales.
    if "full" in data:
        full = np.asarray(data["full"], dtype=np.float64)
        expected = np.diff(levels, axis=1) / scale[:, None]
        if full.ndim != 3 or full.shape[:2] != (n,20) or full.shape[2] < 1:
            raise ValueError("full token shape mismatch.")
        if not np.allclose(expected, full[:,:,0], atol=1e-8, rtol=1e-8):
            raise ValueError("Raw levels, training scale and full increments disagree.")
    if check_target:
        y = np.asarray(data["y"], dtype=np.float64)
        if y.shape != (n,20) or not np.isfinite(y).all():
            raise ValueError("Normalized target must be finite [N,20].")
    else:
        y = None
    return torch.as_tensor(levels), torch.as_tensor(scale), (None if y is None else torch.as_tensor(y))


def fit_patchtst(train, validation, config, seed, max_epochs=60, deadline=None, *, common=None):
    """Return (model-dict, receipt), with the best validated FULL state restored.

    Fixed protocol: float64/thread1 deterministic common.configure; full batch;
    AdamW lr=.001, weight_decay=.001 on all trainable parameters; gradient clip1;
    weighted normalized-target MSE + shrink * weighted prediction^2; max60;
    patience12/min_delta1e-6; epoch0 participates; validation uses common h1/5/20.

    A deadline reached inside the fit returns the restored best state with
    status='deadline' and eligible_for_selection=False. It never begins another
    step after expiry. If expired before entry, raises TimeoutError before fit.
    """
    if not 0 <= int(max_epochs) <= 60:
        raise ValueError("The bounded PatchTST budget is 0..60 epochs.")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("PatchTST deadline reached before fit.")
    common = _common_helpers(common)
    common.configure()
    torch.manual_seed(int(seed))
    started = time.perf_counter()
    if train["market_names"] != validation["market_names"]:
        raise ValueError("Market vocabulary/order differs between train and validation.")
    if not set(np.unique(validation["market"])).issubset(set(np.unique(train["market"]))):
        raise ValueError("Validation includes a market absent from local training.")
    shrink = float(config.get("shrink", 1.0))
    if shrink not in (0.1, 1.0):
        raise ValueError("Prespecified output shrink choices are 0.1 and 1.0.")
    if float(config.get("lr", .001)) != .001 or float(config.get("weight_decay", .001)) != .001:
        raise ValueError("Prespecified PatchTST AdamW lr/weight_decay are both .001.")
    if int(config.get("context_length", 21)) != 21:
        raise ValueError("D2 uses exactly 21 raw price nodes, matching GRU information support.")
    x, scale, y = _inputs(train)
    vx, vscale, _ = _inputs(validation)
    weights = torch.as_tensor(common.group_weights(train), dtype=torch.float64)
    if weights.shape != (len(x),) or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("Invalid equal-market weights.")
    net = LocalPatchTST(d_model=int(config.get("d_model", config.get("hidden",16))),
                        context_length=21, affine=False).double()
    # Preserve author model initialization; no GRU-specific ridge-head init is
    # transplanted into the different PatchTST flatten head.
    net.eval()
    initial_arrays = common.state_arrays(net)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("PatchTST deadline reached before epoch0 validation.")
    with torch.no_grad():
        initial_prediction = net.normalized_delta(vx, vscale)
        best = initial_validation = common.validation_score(validation, initial_prediction)
    if not np.isfinite(best):
        raise FloatingPointError("Nonfinite PatchTST epoch0 validation score.")
    selected, stale = 0, 0
    saved = copy.deepcopy(net.state_dict())
    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.001)
    trace, optimizer_steps = [], 0
    status = "completed"
    stop_reason = "maximum_epochs" if max_epochs else "epoch0_only"
    if deadline is not None and time.monotonic() >= deadline:
        status, stop_reason = "deadline", "after_epoch0_validation"
    for epoch in range(1, (int(max_epochs)+1) if status != "deadline" else 1):
        if deadline is not None and time.monotonic() >= deadline:
            status, stop_reason = "deadline", "before_training_step"
            break
        net.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = net.normalized_delta(x, scale)
        mse = (weights * (prediction-y).square().mean(1)).mean()
        output_shrink = (weights * prediction.square().mean(1)).mean()
        objective = mse + shrink * output_shrink
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("Nonfinite PatchTST objective; no automatic retry.")
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite PatchTST gradient; no automatic retry.")
        optimizer.step()
        optimizer_steps += 1
        row = {"epoch":epoch,"view":"full","objective":float(objective.detach()),
               "supervised":float(mse.detach()),"prediction_penalty":float(output_shrink.detach()),
               "preclip_gradient":float(norm),"validation_score":None,
               "training_mode":"train","validation_mode":"eval"}
        # Do not add evaluation work after the caller's absolute deadline.
        if deadline is not None and time.monotonic() >= deadline:
            row["status"] = "not_validated_deadline"
            trace.append(row)
            status, stop_reason = "deadline", "after_training_step_before_validation"
            break
        net.eval()
        with torch.no_grad():
            val_prediction = net.normalized_delta(vx, vscale)
            score = common.validation_score(validation, val_prediction)
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite PatchTST validation; no automatic retry.")
        row.update(validation_score=score,status="validated")
        trace.append(row)
        if score < best-1e-6:
            best, selected, stale = score, epoch, 0
            # Includes BatchNorm running_mean/running_var/num_batches_tracked.
            saved = copy.deepcopy(net.state_dict())
        else:
            stale += 1
        if deadline is not None and time.monotonic() >= deadline:
            status, stop_reason = "deadline", "after_validation"
            break
        if stale >= 12:
            status, stop_reason = "early_stopped", "patience_12"
            break
    net.load_state_dict(saved)
    net.eval()
    arrays = common.state_arrays(net)
    selected_hash = common.state_hash(arrays)
    actual_config = dict(config,context_length=21,lr=.001,weight_decay=.001,shrink=shrink)
    model = {"kind":"patchtst","net":net,"config":actual_config,
             "market_names":list(train["market_names"]),
             "state":arrays,"state_sha256":selected_hash}
    receipt = {
        "config":actual_config,"seed":int(seed),"status":status,"stop_reason":stop_reason,
        "eligible_for_selection":status in ("completed","early_stopped"),
        "selected_epoch":selected,"epochs":len(trace),"optimizer_steps":optimizer_steps,
        "initial_validation":initial_validation,"validation_score":best,"trace":trace,
        "state_sha256":selected_hash,"initial_state_sha256":common.state_hash(initial_arrays),
        "all_state_dict_entries_saved":True,
        "batchnorm_state_entries":[k for k in arrays if any(t in k for t in ("running_mean","running_var","num_batches_tracked"))],
        "parameter_count":sum(p.numel() for p in net.parameters()),
        "trainable_parameter_count":sum(p.numel() for p in trainable),
        "seconds":time.perf_counter()-started,"maximum_epochs":int(max_epochs),
        "patience":12,"minimum_improvement":1e-6,
        "train_counts":{str(int(m)):int(np.sum(np.asarray(train["market"])==m)) for m in np.unique(train["market"])},
        "local_single_market":len(np.unique(train["market"]))==1,
        "equal_market_weights":True,"full_batch_training":True,
        "prediction_units":"cumulative raw change / training market scale / sqrt(h)",
        "raw_levels_attached_by_caller":True,"epoch0_in_selection":True,
        "author_initialization_retained":True,"gru_ridge_head_initialization_used":False,
        "validation_used_for_early_stopping":True,"model_selection_on_confirmation":False,
        "model_metadata":net.metadata(),
    }
    return model, receipt


def predict_patchtst_normalized(model, data):
    if list(data["market_names"]) != model["market_names"]:
        raise ValueError("Market vocabulary/order changed.")
    levels, scale, _ = _inputs(data, check_target=False)
    model["net"].eval()
    with torch.no_grad():
        return model["net"].normalized_delta(levels, scale).cpu().numpy()


def predict_patchtst(model, data):
    return predict_patchtst_normalized(model,data) * np.asarray(data["scale"])[:,None] * np.sqrt(np.arange(1,21))


def serialize_patchtst_state(model, *, common=None):
    """Full model state, including BatchNorm buffers; use caller's state hash."""
    return _common_helpers(common).state_arrays(model["net"])
