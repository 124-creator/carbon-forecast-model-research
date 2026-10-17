"""Actual bounded CPU fitting; shared training rules and auditable saved states."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn

from data_core import group_weights, metrics
from information_model import SharedInformationGRU, fixed_coarse_features, rbf_mixture


def configure():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def network_input(data, view, price_only=False):
    """Market is observable in BOTH information sets. Always pass the mask:
    the final channel here is a market indicator, not a validity indicator.
    """
    x = data[view][..., :1] if price_only else data[view]
    identity = np.eye(len(data["market_names"]))[data["market"]]
    identity = np.broadcast_to(identity[:, None, :], (len(x), x.shape[1], identity.shape[-1]))
    identity = identity * data[view + "_mask"][..., None]
    return torch.as_tensor(np.concatenate([x, identity], axis=-1))


class PlainLSTM(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.gru = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 20)

    def forward(self, x, mask=None):
        lengths = mask.sum(1) if mask is not None else torch.full((len(x),), x.shape[1])
        if not torch.all(lengths == lengths[0]):
            raise ValueError("Prespecified LSTM views have a common length")
        out, _ = self.gru(x[:, :int(lengths[0])])
        return self.head(out[:, -1])


def build_network(config, markets):
    price_only = config["family"] in ("gru_price", "lstm_price", "gru_stat_price")
    channels = (1 if price_only else 3) + markets
    if config["family"].startswith("lstm"):
        net = PlainLSTM(channels, config["hidden"])
    else:
        net = SharedInformationGRU(channels, config["hidden"], 20)
    return net


def head_features(net, x):
    out, _ = net.gru(x)
    z = out[:, -1]
    return torch.cat([z, torch.ones_like(z[:, :1])], dim=1)


def initialize_head(net, x, y, weights, shrink):
    """Same training-only ridge initialization for every neural intervention."""
    with torch.no_grad():
        z = head_features(net, x)
        weighted_z = z * weights[:, None]
        a = (1.0 + shrink) * (z.T @ weighted_z) / len(z) + .01 * torch.eye(z.shape[1])
        b = z.T @ (weights[:, None] * y) / len(z)
        beta = torch.linalg.solve(a, b)
        net.head.weight.copy_(beta[:-1].T)
        net.head.bias.copy_(beta[-1])


def make_moment_cache(train, view, blocked=True):
    """Exact blockwise computation of the candidate's fixed kernel objective.
    No full B*B*feature-dimensional temporary tensor is formed.
    """
    c = fixed_coarse_features(torch.as_tensor(train[view]))
    caches, audit = [], []
    for market in np.unique(train["market"]):
        ids = np.flatnonzero(train["market"] == market)
        origin = torch.as_tensor(train["origin"][ids])
        if blocked:
            eligible = torch.abs(origin[:, None] - origin[None, :]) > 40
        else:
            eligible = ~torch.eye(len(ids), dtype=torch.bool)
        count = int(eligible.sum())
        if not count:
            raise ValueError("A market has no eligible conditional-moment pairs")
        k = rbf_mixture(c[ids])
        weights = k * eligible / count
        caches.append((torch.as_tensor(ids), weights))
        audit.append(dict(market=int(market), view=view, ordered_pairs=count,
                          minimum_origin_separation=41 if blocked else 1,
                          kernel_weight_sum=float((k * eligible).sum()),
                          fixed_feature_dimension=c.shape[1]))
    return caches, audit


def cached_moment(fine, coarse, caches):
    d = fine - coarse
    raw = torch.stack([(d[ids] * (w @ d[ids])).sum(0) for ids, w in caches])
    value = raw.clamp_min(0).mean()
    return value, float((raw.detach() > 0).double().mean())


def validation_score(data, prediction):
    """Same selection metric for every intervention: equal markets and h1/5/20."""
    records = []
    horizons = torch.as_tensor([0, 4, 19])
    y = torch.as_tensor(data["y"])
    for mid in np.unique(data["market"]):
        use = torch.as_tensor(data["market"] == mid)
        mse = (prediction[use][:, horizons] - y[use][:, horizons]).square().mean(0)
        rw = y[use][:, horizons].square().mean(0)
        records.append((mse / rw).mean())
    return float(torch.stack(records).mean())


def state_arrays(net):
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


def state_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def fit(train, validation, config, seed, max_epochs=60, deadline=None):
    configure()
    torch.manual_seed(seed)
    started = time.perf_counter()
    family = config["family"]
    weights = torch.as_tensor(group_weights(train))
    y = torch.as_tensor(train["y"])
    use_anchor = family == "gru_stat_price"
    if use_anchor:
        y = y - torch.as_tensor(train["stat_normalized"])
    if family == "ridge":
        x = train["full"].reshape(len(y), -1)
        ids = np.eye(len(train["market_names"]))[train["market"]]
        z = np.c_[x, ids]
        w = weights.numpy()
        alpha = float(config["alpha"])
        beta = np.linalg.solve(z.T @ (z * w[:, None]) / len(z) + alpha * np.eye(z.shape[1]),
                               z.T @ (train["y"] * w[:, None]) / len(z))
        model = dict(kind="ridge", beta=beta, config=dict(config), market_names=train["market_names"])
        val = predict_normalized(model, validation, "full")
        receipt = dict(config=dict(config), seed=seed, parameter_count=beta.size, selected_epoch=0,
                       epochs=0, validation_score=validation_score(validation, torch.as_tensor(val)),
                       trace=[], state_sha256=state_hash({"beta": beta}), seconds=time.perf_counter() - started)
        return model, receipt

    price_only = family in ("gru_price", "lstm_price", "gru_stat_price")
    net = build_network(config, len(train["market_names"]))
    x = {view: network_input(train, view, price_only) for view in ("full", "short", "sparse")}
    masks = {view: torch.as_tensor(train[view + "_mask"]) for view in x}
    vx = network_input(validation, "full", price_only)
    vm = torch.as_tensor(validation["full_mask"])
    shrink = float(config.get("shrink", 1.0))
    initialize_head(net, x["full"], y, weights, shrink)
    initial_arrays = state_arrays(net)
    with torch.no_grad():
        initial_pred = net(vx, mask=vm)
        if use_anchor:
            initial_pred = initial_pred + torch.as_tensor(validation["stat_normalized"])
        best = initial_validation = validation_score(validation, initial_pred)
    selected, stale = 0, 0
    saved = copy.deepcopy(net.state_dict())
    optimizer = torch.optim.AdamW([
        dict(params=net.gru.parameters(), weight_decay=.001),
        dict(params=net.head.parameters(), weight_decay=0.)], lr=float(config.get("lr", .003)))
    augmented = family in ("augment", "kcm", "pointwise", "kcm_unblocked")
    cached, kernel_audit = {}, []
    if family in ("kcm", "kcm_unblocked"):
        for view in ("short", "sparse"):
            cached[view], a = make_moment_cache(train, view, family == "kcm")
            kernel_audit.extend(a)
    trace = []
    for epoch in range(1, max_epochs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Frozen fitting deadline reached")
        optimizer.zero_grad(set_to_none=True)
        fine = net(x["full"], mask=masks["full"])
        mse = (weights * (fine - y).square().mean(1)).mean()
        shrink_loss = (weights * fine.square().mean(1)).mean()
        penalty = fine.sum() * 0
        positive_fraction = 0.0
        view = "short" if epoch % 2 else "sparse"
        if augmented:
            coarse = net(x[view], mask=masks[view])
            mse = .5 * (mse + (weights * (coarse - y).square().mean(1)).mean())
            shrink_loss = .5 * (shrink_loss + (weights * coarse.square().mean(1)).mean())
            if family in ("kcm", "kcm_unblocked"):
                penalty, positive_fraction = cached_moment(fine, coarse, cached[view])
            elif family == "pointwise":
                penalty = (weights * (fine - coarse).square().mean(1)).mean()
        head_l2 = .01 * (net.head.weight.square().sum() + net.head.bias.square().sum()) / 20
        objective = mse + shrink * shrink_loss + head_l2 + float(config.get("strength", 0)) * penalty
        objective.backward()
        gradient = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            val_pred = net(vx, mask=vm)
            if use_anchor:
                val_pred = val_pred + torch.as_tensor(validation["stat_normalized"])
            val = validation_score(validation, val_pred)
        row = dict(epoch=epoch, view=view if augmented else "full", objective=float(objective.detach()),
                   supervised=float(mse.detach()), prediction_penalty=float(shrink_loss.detach()),
                   mechanism_penalty=float(penalty.detach()), positive_moment_fraction=positive_fraction,
                   preclip_gradient=float(gradient), validation_score=val)
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise FloatingPointError("Nonfinite training trace; no automatic retry")
        trace.append(row)
        if val < best - 1e-6:
            best, selected, stale = val, epoch, 0
            saved = copy.deepcopy(net.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    net.load_state_dict(saved)
    arrays = state_arrays(net)
    model = dict(kind="neural", net=net, config=dict(config), market_names=train["market_names"])
    receipt = dict(config=dict(config), seed=int(seed), parameter_count=sum(p.numel() for p in net.parameters()),
                   selected_epoch=selected, epochs=len(trace), initial_validation=initial_validation,
                   validation_score=best, trace=trace, state_sha256=state_hash(arrays),
                   initial_state_sha256=state_hash(initial_arrays), kernel_audit=kernel_audit,
                   seconds=time.perf_counter() - started,
                   train_counts={name: int((train["market"] == i).sum()) for i, name in enumerate(train["market_names"])},
                   input_includes_market_identity=True, masks_passed_explicitly=True,
                   validation_used_for_early_stopping=True, model_selection_on_confirmation=False)
    return model, receipt


def predict_normalized(model, data, view="full"):
    if data["market_names"] != model["market_names"]:
        raise ValueError("Market ID order changed")
    if model["kind"] == "ridge":
        x = data[view].reshape(len(data[view]), -1)
        ids = np.eye(len(data["market_names"]))[data["market"]]
        return np.c_[x, ids] @ model["beta"]
    price_only = model["config"]["family"] in ("gru_price", "lstm_price", "gru_stat_price")
    with torch.no_grad():
        pred = model["net"](network_input(data, view, price_only),
                              mask=torch.as_tensor(data[view + "_mask"])).numpy()
        if model["config"]["family"] == "gru_stat_price":
            if view != "full":
                raise ValueError("Full-history statistical anchor has no coarse-view claim")
            pred = pred + data["stat_normalized"]
        return pred


def predict(model, data, view="full"):
    return predict_normalized(model, data, view) * data["scale"][:, None] * np.sqrt(np.arange(1, 21))


def serialize_state(model):
    return state_arrays(model["net"]) if model["kind"] == "neural" else {"beta": model["beta"]}


def moment_implementation_check():
    """Independent public loss vs cached algebra on finite artificial examples."""
    from information_model import blocked_kernel_moment_penalty
    configure(); torch.manual_seed(991)
    n = 24
    x = torch.randn(n, 20, 3)
    pf, pc = torch.randn(n, 20, requires_grad=True), torch.randn(n, 20, requires_grad=True)
    origin = np.r_[np.arange(12) * 23 + 20, np.arange(12) * 23 + 20]
    mid = np.repeat([0, 1], 12)
    data = {"short": x.numpy(), "origin": origin, "market": mid}
    caches, _ = make_moment_cache(data, "short")
    fast, _ = cached_moment(pf, pc, caches)
    slow, _ = blocked_kernel_moment_penalty(pf, pc, fixed_coarse_features(x),
                                           torch.as_tensor(origin - 20), torch.as_tensor(origin + 20),
                                           groups=torch.as_tensor(mid))
    gf = torch.autograd.grad(fast, (pf, pc), retain_graph=True)
    gs = torch.autograd.grad(slow, (pf, pc))
    err = max(float(abs(fast - slow)), *(float((a - b).abs().max()) for a, b in zip(gf, gs)))
    assert err < 1e-12, err
    return dict(passed=True, objective_and_gradient_max_difference=err, market_fits=0)
