"""Fresh fits only; common point objective and checkpoint policy."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import copy
import time
import numpy as np
import torch
from fair_core import RecurrentNet, configure, reference
from fair_data import support

def fit(price, end, mode, config, seed, max_epochs=60, deadline=None):
    configure()
    torch.manual_seed(int(seed))
    prefix = np.array(price[:end], copy=True)
    origins, tr, va, scale, target = support(prefix, end)
    base, context = reference(prefix)
    model = RecurrentNet(mode, int(config['hidden']))
    residual = torch.as_tensor((target - base[origins]) / scale)
    with torch.no_grad():
        model.initialize(model.features(prefix, context, origins[tr], scale), residual[tr])
    encoder_parameters = [p for name, p in model.named_parameters() if name != 'readout']
    groups = [dict(params=encoder_parameters, weight_decay=float(config['weight_decay']))]
    if isinstance(model.readout, torch.nn.Parameter):
        groups.append(dict(params=[model.readout], weight_decay=0.))
    optimizer = torch.optim.AdamW(groups, lr=float(config['lr']))
    params = list(model.parameters())

    def validation_loss():
        with torch.no_grad():
            g = model.features(prefix, context, origins[va], scale)
            return float((model.correction(g) - residual[va]).square().mean())

    best = validation_loss()
    initial = best
    selected = 0
    stale = 0
    saved = copy.deepcopy(model.state_dict())
    trace = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Frozen fit budget exhausted')
        optimizer.zero_grad(set_to_none=True)
        g = model.features(prefix, context, origins[tr], scale)
        if mode == 'gru_profile':
            model.initialize(g.detach(), residual[tr])
        correction = model.correction(g)
        expanded = model.readout @ model.basis.T
        loss = ((correction - residual[tr]).square() + correction.square()).mean() + .01 * expanded.square().sum() / 20
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(params, 1.))
        optimizer.step()
        if mode == 'gru_profile':
            with torch.no_grad():
                model.initialize(model.features(prefix, context, origins[tr], scale), residual[tr])
        val = validation_loss()
        row = [epoch, float(loss.detach()), val, gradient]
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
    receipt = dict(family=mode, config=dict(config), seed=int(seed), end=int(end),
                   scale=scale, train_n=len(tr), validation_n=len(va),
                   train_max_target=int(origins[tr[-1]] + 20), validation_first_origin=int(origins[va[0]]),
                   validation_max_target=int(origins[va[-1]] + 20), selected_epoch=selected,
                   epochs=len(trace), initial_validation=initial, best_validation=best,
                   learned_parameters=sum(p.numel() for p in model.parameters()),
                   readout_coefficients=model.readout.numel(),
                   seconds=time.perf_counter() - started, trace=trace,
                   trace_columns=['epoch', 'common_objective', 'validation_MSE', 'preclip_gradient'])
    return model, scale, receipt

def predict(model, scale, price, origins):
    origins = np.asarray(origins, dtype=np.int64)
    prefix = np.asarray(price[:int(origins.max()) + 1])
    base, context = reference(prefix)
    with torch.no_grad():
        correction = model.correction(model.features(prefix, context, origins, scale)).numpy()
    return base[origins] + scale * correction
