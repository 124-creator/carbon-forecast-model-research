"""Fresh full-GRU fits and strictly observed residual-memory prediction."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import copy
import hashlib
import time
import numpy as np
import torch
from rum_core import MemoryNet, candidate_cache, utility_teacher
from fair_core import configure, reference
from fair_data import support


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def inputs(price, ids, scale):
    base, context = reference(price)
    cache = candidate_cache(price, ids)
    residual_values = np.zeros((*cache['index'].shape, 20))
    rows, columns = np.nonzero(cache['valid'])
    keys = ids[cache['index'][rows, columns]]
    if len(keys) and np.any(keys + 20 > ids[rows] - 60):
        raise ValueError('Unmatured/overlapping memory value')
    if len(keys):
        values = price[keys[:, None] + np.arange(1, 21)] - price[keys, None]
        residual_values[rows, columns] = (values - base[keys]) / scale
    return base, context, {k: torch.as_tensor(v) for k, v in cache.items()}, torch.as_tensor(residual_values)


def fit(price, end, family, seed, max_epochs=60, deadline=None):
    configure()
    torch.manual_seed(int(seed))
    prefix = np.array(price[:end], copy=True)
    ids, train, validation, scale, target = support(prefix, end)
    base, context, cache, memory_values = inputs(prefix, ids, scale)
    model = MemoryNet(family)
    model.fit_end = int(end)
    model.pool_start = int(ids[0])
    residual = torch.as_tensor((target - base[ids]) / scale)
    horizon_variance = torch.as_tensor(np.maximum(np.var(target[train] / scale, axis=0), 1e-6))
    with torch.no_grad():
        model.initialize(model.features(prefix, context, ids[train], scale), residual[train])
    groups = [dict(params=[p for name, p in model.named_parameters() if name != 'readout' and p.requires_grad], weight_decay=.001),
              dict(params=[model.readout], weight_decay=0.)]
    optimizer = torch.optim.AdamW(groups, lr=.003)
    parameters = [p for p in model.parameters() if p.requires_grad]

    def forward():
        return model.memory_forward(model.features(prefix, context, ids, scale), cache, memory_values)

    def validation_loss():
        with torch.no_grad():
            correction, _, _, _, _ = forward()
            return float((correction[validation] - residual[validation]).square().mean())

    started = time.perf_counter()
    best = initial = validation_loss()
    saved = copy.deepcopy(model.state_dict())
    selected, stale, trace = 0, 0, []
    for epoch in range(1, max_epochs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Frozen memory pilot deadline passed')
        optimizer.zero_grad(set_to_none=True)
        correction, plain, memory, logits, weights = forward()
        expanded = model.readout @ model.basis.T
        common = ((correction[train] - residual[train]).square() + correction[train].square()).mean()
        common = common + .01 * expanded.square().sum() / 20
        auxiliary = torch.zeros(())
        if family == 'utility':
            teacher = utility_teacher(residual[train] - plain[train], memory[train], cache['valid'][train], horizon_variance)
            auxiliary = (teacher * (teacher.clamp_min(1e-12).log() - torch.log_softmax(logits[train], dim=1))).sum(1).mean()
        loss = common + .1 * auxiliary
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
        optimizer.step()
        val = validation_loss()
        row = [epoch, float(common.detach()), float(auxiliary.detach()), val, gradient]
        if not np.isfinite(row).all():
            raise FloatingPointError('Nonfinite fit; no hidden retry')
        trace.append(row)
        if val < best - 1e-6:
            best, selected, stale = val, epoch, 0
            saved = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(saved)
    receipt = dict(family=family, seed=int(seed), end=int(end), scale=scale,
                   train_n=len(train), validation_n=len(validation), pool_start=int(ids[0]),
                   train_max_target=int(ids[train[-1]] + 20), validation_first_origin=int(ids[validation[0]]),
                   validation_max_target=int(ids[validation[-1]] + 20), horizon_variance=horizon_variance.tolist(),
                   epochs=len(trace), selected_epoch=selected, initial_validation=initial, best_validation=best,
                   allocated_parameters=sum(p.numel() for p in model.parameters()),
                   active_parameters=sum(p.numel() for p in parameters), state_sha256=state_digest(model),
                   seconds=time.perf_counter() - started, trace=trace,
                   trace_columns=['epoch', 'common_objective', 'utility_KL', 'validation_MSE', 'preclip_gradient'])
    return model, scale, receipt


def predict(model, scale, price, origins):
    origins = np.asarray(origins, dtype=np.int64)
    if not len(origins) or origins.min() < model.pool_start or origins.max() >= len(price):
        raise ValueError('Invalid forecast support')
    prefix = np.array(price[:int(origins.max()) + 1], copy=True)
    ids = np.arange(model.pool_start, int(origins.max()) + 1)
    base, context, cache, memory_values = inputs(prefix, ids, scale)
    with torch.no_grad():
        correction, plain, _, _, weights = model.memory_forward(model.features(prefix, context, ids, scale), cache, memory_values)
        where = origins - model.pool_start
        forecasts = base[origins] + scale * correction[where].numpy()
        diagnostics = dict(mean_null_weight=float(weights[where, 0].mean()),
                           maximum_memory_weight=float(weights[where, 1:].max()),
                           mean_memory_square=float((correction[where] - plain[where]).square().mean()))
    return forecasts, diagnostics
