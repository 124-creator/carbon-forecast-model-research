"""Fresh candidate/control fitting with shared mature-label support."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import copy
import sys
import time
from pathlib import Path
import numpy as np
import torch
from hsc_core import ScaleNet, configure, training_loss
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fair_recurrent_tuning_20261016'))
from fair_core import reference
from fair_data import support


def fit(price, end, mode, config, seed, max_epochs=60, deadline=None):
    configure()
    torch.manual_seed(int(seed))
    prefix = np.array(price[:end], copy=True)
    origins, train, validation, scale, target = support(prefix, end)
    base, context = reference(prefix)
    model = ScaleNet(mode, int(config['hidden']))
    residual = torch.as_tensor((target - base[origins]) / scale)
    with torch.no_grad():
        model.initialize(model.features(prefix, context, origins[train], scale), residual[train])
    groups = [dict(params=[p for n, p in model.named_parameters() if n != 'readouts'],
                   weight_decay=float(config['weight_decay'])),
              dict(params=[model.readouts], weight_decay=0.)]
    optimizer = torch.optim.AdamW(groups, lr=float(config['lr']))

    def validation_loss():
        with torch.no_grad():
            out = model(prefix, context, origins[validation], scale)
            return float((out['prediction'] - residual[validation]).square().mean())

    best = initial = validation_loss()
    selected = stale = 0
    saved = copy.deepcopy(model.state_dict())
    trace = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Frozen fit budget exhausted')
        optimizer.zero_grad(set_to_none=True)
        out = model(prefix, context, origins[train], scale)
        loss, parts = training_loss(model, out, residual[train])
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        val = validation_loss()
        row = [epoch, float(loss.detach()), val, gradient, parts['main'], parts['auxiliary'], parts['competition']]
        if not np.isfinite(row).all():
            raise FloatingPointError('Nonfinite fit; no retry')
        trace.append(row)
        if val < best - 1e-6:
            best, selected, stale = val, epoch, 0
            saved = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(saved)
    receipt = dict(family=mode, config=dict(config), seed=int(seed), end=int(end), scale=scale,
                   train_n=len(train), validation_n=len(validation),
                   train_max_target=int(origins[train[-1]] + 20),
                   validation_first_origin=int(origins[validation[0]]), validation_max_target=int(origins[-1] + 20),
                   epochs=len(trace), selected_epoch=selected, initial_validation=initial, best_validation=best,
                   parameters=sum(p.numel() for p in model.parameters()), scales=list(model.scales),
                   seconds=time.perf_counter() - started, trace=trace,
                   trace_columns=['epoch', 'objective', 'validation_MSE', 'preclip_gradient',
                                  'main_risk', 'expert_risk', 'competition_CE'])
    return model, scale, receipt


def predict(model, scale, price, origins):
    ids = np.asarray(origins, dtype=np.int64)
    prefix = np.asarray(price[:int(ids.max()) + 1])
    base, context = reference(prefix)
    with torch.no_grad():
        out = model(prefix, context, ids, scale)
    return base[ids] + scale * out['prediction'].numpy(), out['weights'].numpy()
