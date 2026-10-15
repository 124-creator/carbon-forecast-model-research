"""Fresh complete GRU training; no old predictions, weights or teacher."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import copy
import time
import numpy as np
import torch
from torch import nn
from event_inputs import make_arrays
from event_model import EventNet
from reference_support import configure, reference, support


def tensors(data, end, train_last, scale, stats=None):
    arrays, stats = make_arrays(data['price'], data['dates'], data['source_dates'],
                               data['source_values'], end, train_last, scale, stats)
    base, context = reference(data['price'][:end])
    context[:, 0] = np.tanh(context[:, 0] / scale)
    return ({k: torch.as_tensor(v) for k, v in arrays.items()},
            torch.as_tensor(context), torch.as_tensor(base / scale)), stats


def fit(data, end, mode, seed, max_epochs=60, deadline=None):
    configure()
    torch.manual_seed(int(seed))
    prefix = np.array(data['price'][:end], copy=True)
    origins, tr, va, scale, target = support(prefix, end)
    args, stats = tensors(data, end, int(origins[tr[-1]]), scale)
    ids = torch.as_tensor(origins, dtype=torch.long)
    tr_ids, va_ids = ids[tr], ids[va]
    truth = torch.as_tensor(target / scale)
    model = EventNet(mode)

    def install():
        with torch.no_grad():
            features = model.features(args[0], args[1], tr_ids)
            residual = truth[tr] - args[2][tr_ids]
            n, d = features.shape
            fitted = torch.linalg.solve(2 * features.T @ features / n + .01 * torch.eye(d),
                                        features.T @ residual / n)
            model.B.copy_(fitted)

    def validation():
        with torch.no_grad():
            return float((model(*args, va_ids) - truth[va]).square().mean())

    install()
    initial = best = validation()
    best_state = copy.deepcopy(model.state_dict())
    chosen = stale = 0
    trace = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    for epoch in range(1, max_epochs + 1):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError('Frozen source-learning market budget exhausted')
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*args, tr_ids)
        correction = prediction - args[2][tr_ids]
        loss = ((prediction - truth[tr]).square() + correction.square()).mean() + .01 * model.B.square().sum() / 20
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        install()
        val = validation()
        row = [epoch, float(loss.detach()), val, norm]
        if not np.isfinite(row).all():
            raise FloatingPointError('Nonfinite training; no rescue tuning')
        trace.append(row)
        if val < best - 1e-6:
            best, chosen, stale = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(best_state)
    stats = dict(stats, train_last=int(origins[tr[-1]]))
    log = dict(mode=mode, seed=int(seed), end=int(end), train_n=len(tr), validation_n=len(va),
               scale=scale, train_max_target=int(origins[tr[-1]] + 20),
               validation_first_origin=int(origins[va[0]]), validation_max_target=int(origins[va[-1]] + 20),
               parameters=sum(p.numel() for p in model.parameters()), analytic_head_values=model.B.numel(),
               source_fit_reports=stats['fit_reports'], source_fit_date=stats['fit_date'],
               initial_validation=initial, best_validation=best, selected_epoch=chosen,
               epochs=len(trace), history_columns=['epoch', 'objective', 'validation_MSE', 'gradient_norm_preclip'],
               history=trace, stats=stats)
    return model, stats, log


def predict(model, stats, data, origins):
    origins = np.asarray(origins, dtype=np.int64)
    args, _ = tensors(data, int(origins.max()) + 1, stats['train_last'], stats['scale'], stats)
    with torch.no_grad():
        return (model(*args, torch.as_tensor(origins, dtype=torch.long)) * stats['scale']).numpy()
