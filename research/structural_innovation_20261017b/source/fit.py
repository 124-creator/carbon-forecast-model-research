"""One bounded annual fit under a common native-price prediction protocol."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import copy
import hashlib
import time

import numpy as np
import torch

from data import metrics, risk_denominator
from models import make_model, auxiliary_weight
from smp_model import hurdle_nll


def configure():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def state_arrays(net):
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


def state_hash(net):
    h = hashlib.sha256()
    for name, value in sorted(state_arrays(net).items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def tensor_batch(batch):
    return {k: torch.as_tensor(v, dtype=torch.long if k in ("state", "event", "origin") else torch.float64)
            for k, v in batch.items() if k in ("x", "scale", "y", "delta", "state", "event", "origin")}


def objective(family, mean, aux, batch, denominator, shrink):
    """Ordinary risk alignment is a baseline correction, not SMP innovation."""
    use = slice(0, 1) if family == "smp_one_step" else slice(None)
    if family == "gru_normalized":
        hh = torch.arange(1, 21, dtype=mean.dtype, device=mean.device)
        supervised = ((mean - batch["y"]).square() / hh).mean()
        penalty = (mean.square() / hh).mean()
    else:
        raw_mean = mean[:, use] * batch["scale"][:, None]
        supervised = ((raw_mean - batch["delta"][:, use]).square() / denominator[use]).mean()
        penalty = (raw_mean.square() / denominator[use]).mean()
    aw = auxiliary_weight(family)
    likelihood = (hurdle_nll(aux, batch["y"][:, 0], reduction="mean", event=batch["event"] != 0)
                  if aw else mean.new_zeros(()))
    loss = supervised + float(shrink) * penalty + aw * likelihood
    return loss, supervised, penalty, likelihood


def predict(net, batch, chunk_size=256):
    net.eval()
    out = []
    with torch.no_grad():
        for left in range(0, len(batch["x"]), chunk_size):
            sl = slice(left, left + chunk_size)
            x = torch.as_tensor(batch["x"][sl], dtype=torch.float64)
            state = torch.as_tensor(batch["state"][sl], dtype=torch.long)
            out.append(net(x, state).cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_neural(train, validation, family, shrink, seed, *, max_epochs=50, patience=10,
               batch_size=128, deadline=None, min_delta=1e-6):
    configure()
    torch.manual_seed(int(seed))
    started = time.perf_counter()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Fit deadline expired before initialization")
    if max_epochs > 50 or max_epochs < 0 or patience != 10 or batch_size != 128:
        raise ValueError("Outside the frozen implementation budget")
    q0 = float(np.mean(np.asarray(train["event"]) == 0))
    nonzero = np.asarray(train["event"]) != 0
    mark_scale = float(np.mean(abs(train["y"][nonzero, 0]))) if nonzero.any() else 1.
    net = make_model(family, train["x"].shape[-1], q0, mark_scale).double()
    x = tensor_batch(train)
    denominator = torch.as_tensor(risk_denominator(train))
    learning_rate = .001 if family == "patchtst_native" else .003
    optim = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=.001)
    generator = torch.Generator().manual_seed(int(seed) + 1000003)
    initial_hash = state_hash(net)
    initial_prediction = predict(net, validation)
    initial_score = metrics(initial_prediction * validation["scale"][:, None], validation)["score"]
    if not np.isfinite(initial_score):
        raise FloatingPointError("Nonfinite epoch0 validation")
    best_score, selected_epoch, stale = initial_score, 0, 0
    saved = copy.deepcopy(net.state_dict())
    trace = []
    steps = 0
    stop_reason, status = "maximum_epochs", "completed"
    for epoch in range(1, max_epochs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            status, stop_reason = "deadline", "before_epoch"
            break
        net.train()
        order = torch.randperm(len(train["x"]), generator=generator)
        weighted = np.zeros(4)
        max_grad = 0.
        seen = 0
        for left in range(0, len(order), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                status, stop_reason = "deadline", "before_batch"
                break
            ids = order[left:left + batch_size]
            batch = {k: v[ids] for k, v in x.items()}
            optim.zero_grad(set_to_none=True)
            mean, aux = net(batch["x"], batch["state"], return_aux=True)
            losses = objective(family, mean, aux, batch, denominator, shrink)
            loss = losses[0]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite objective; no automatic retry")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            if not bool(torch.isfinite(grad)):
                raise FloatingPointError("Nonfinite gradient; no automatic retry")
            optim.step()
            steps += 1
            seen += len(ids)
            weighted += len(ids) * np.array([float(v.detach()) for v in losses])
            max_grad = max(max_grad, float(grad))
        row = dict(epoch=epoch, training_samples_seen=seen, optimizer_steps_total=steps,
                   objective=float(weighted[0] / max(seen, 1)), supervised=float(weighted[1] / max(seen, 1)),
                   output_penalty=float(weighted[2] / max(seen, 1)), hurdle_nll=float(weighted[3] / max(seen, 1)),
                   maximum_preclip_gradient=max_grad, validation_score=None)
        if status != "completed":
            trace.append(row)
            break
        if deadline is not None and time.monotonic() >= deadline:
            status, stop_reason = "deadline", "before_validation"
            trace.append(row)
            break
        vp = predict(net, validation)
        score = metrics(vp * validation["scale"][:, None], validation)["score"]
        row["validation_score"] = score
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation; no automatic retry")
        trace.append(row)
        if score < best_score - min_delta:
            best_score, selected_epoch, stale = score, epoch, 0
            saved = copy.deepcopy(net.state_dict())
        else:
            stale += 1
        if stale >= patience:
            stop_reason = "validation_patience"
            break
    net.load_state_dict(saved)
    net.eval()
    final_validation = metrics(predict(net, validation) * validation["scale"][:, None], validation)["score"]
    if abs(final_validation - best_score) > 1e-10:
        raise AssertionError("Restored full checkpoint differs from validated issued predictor")
    receipt = dict(family=family, seed=int(seed), shrink=float(shrink), status=status,
                   eligible_for_selection=status == "completed", stop_reason=stop_reason,
                   parameters=sum(p.numel() for p in net.parameters()),
                   trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
                   initial_state_sha256=initial_hash, selected_state_sha256=state_hash(net),
                   initial_validation_score=initial_score, selected_validation_score=best_score,
                   selected_epoch=selected_epoch, executed_epochs=len(trace), optimizer_steps=steps,
                   wall_seconds=time.perf_counter() - started, trace=trace,
                   learning_rate=learning_rate, weight_decay=.001, gradient_clip=1.,
                   auxiliary_nll_weight=auxiliary_weight(family), native_horizons=[1] if family == "smp_one_step" else list(range(1, 21)),
                   primary_training_objective="local-scale/sqrt(h) normalized MSE" if family == "gru_normalized" else "native price MSE / training-label RW MSE by horizon",
                   best_checkpoint_uses_actual_final_output=True, epoch_zero_eligible=True,
                   risk_denominator_sha256=hashlib.sha256(denominator.cpu().numpy().tobytes()).hexdigest(),
                   minibatch_order="same seeded permutation for every neural family/config within market/year",
                   training_mode="train", evaluation_mode="eval", dtype="float64", cpu_threads=1)
    return net, receipt
