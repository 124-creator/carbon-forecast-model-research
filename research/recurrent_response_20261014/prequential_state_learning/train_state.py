"""Chronological recurrent training; source-independent NumPy/Torch implementation."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import copy
import numpy as np
import torch
from recurrent_state import RecurrentPredictor, profile_head, prequential_loss

MODES = ('window_joint', 'full_joint', 'window_profile', 'full_profile',
         'full_prequential_complete', 'full_prequential_mature')
SEEDS = (17, 29)


def configure():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)


def reference(price):
    bank = [(0., 1.)] + [(r, q) for r in (0., .25, .5, .75, .9, .97) for q in (.02, .1, .25, .5, .8)]
    rho, permanent = np.array(bank).T
    transitory = 1 - permanent
    n = len(price)
    state = np.zeros(len(bank))
    variance = transitory / (1 - rho ** 2)
    states = np.zeros((n, len(bank)))
    errors = np.zeros_like(states)
    innovations = np.ones_like(states)
    for t in range(1, n):
        error = price[t] - price[t - 1] - (rho - 1) * state
        innovation = (rho - 1) ** 2 * variance + 1
        covariance = rho * (rho - 1) * variance + transitory
        state = rho * state + covariance * error / innovation
        variance = np.maximum(rho ** 2 * variance + transitory - covariance ** 2 / innovation, 0)
        states[t], errors[t], innovations[t] = state, error, innovation
    base = np.zeros((n, 20))
    context = np.zeros((n, 3))
    selected = 0
    for t in range(n):
        if t >= 60 and (t - 60) % 20 == 0:
            start = max(1, t - 240)
            scale = np.maximum(np.mean(errors[start:t] ** 2 / innovations[start:t], axis=0), 1e-12)
            criterion = np.log(innovations[start:t]).sum(0) + (t - start) * np.log(scale)
            selected = int(np.argmin(criterion))
        r, q = bank[selected]
        context[t] = [states[t, selected], r * (1 - q), 1 - q]
        base[t] = (r ** np.arange(1, 21) - 1) * states[t, selected]
    return base, context


def support(price, end):
    origins = np.arange(max(60, end - 20 - 480), end - 20)
    boundary = len(origins) - 60
    train = np.arange(boundary - 20)
    validation = np.arange(boundary, len(origins))
    if len(train) < 60 or origins[train[-1]] + 20 >= origins[validation[0]] or origins[-1] + 20 >= end:
        raise ValueError('Maturity or purging violation')
    windows = np.stack([np.diff(price[t - 60:t + 1]) for t in origins])
    scale = max(float(windows[train].std()), 1e-6)
    target = np.array([price[t + 1:t + 21] - price[t] for t in origins])
    return origins, train, validation, scale, target


def inputs(price, base, context, origins, scale):
    q = context[origins].copy()
    q[:, 0] = np.tanh(q[:, 0] / scale)
    return (torch.as_tensor(np.diff(price) / scale), torch.as_tensor(q),
            torch.as_tensor(origins, dtype=torch.long), torch.as_tensor(base[origins] / scale))


def fit(price, end, mode, seed, max_epochs=80):
    if mode not in MODES:
        raise ValueError(mode)
    configure()
    torch.manual_seed(seed)
    model = RecurrentPredictor()
    prefix = np.array(price[:end], copy=True)
    base, context = reference(prefix)
    origins, tr, va, scale, target = support(prefix, end)
    delta, q, ot, b = inputs(prefix, base, context, origins, scale)
    residual = torch.as_tensor(target / scale) - b
    history_kind = mode.split('_')[0]
    with torch.no_grad():
        initial_features = model.features(delta, q[tr], ot[tr], history_kind)
        model.B.copy_(profile_head(initial_features, residual[tr]))
    joint = mode.endswith('_joint')
    params = list(model.parameters()) if joint else [p for name, p in model.named_parameters() if name != 'B']
    optimizer = torch.optim.AdamW(params, lr=.003, weight_decay=.001)
    def validation_loss():
        with torch.no_grad():
            g = model.features(delta, q[va], ot[va], history_kind)
            return float((g @ model.B - residual[va]).square().mean())
    best = validation_loss()
    initial_validation = best
    best_state = copy.deepcopy(model.state_dict())
    selected = 0
    stale = 0
    trace = []
    query_start = max(0, len(tr) - 96)
    queries = torch.as_tensor(np.unique(np.linspace(query_start, len(tr) - 1, min(32, len(tr) - query_start)).astype(int)))
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        g = model.features(delta, q[tr], ot[tr], history_kind)
        if 'prequential' in mode:
            loss = prequential_loss(g, residual[tr], ot[tr], queries, complete=mode.endswith('_complete'))
        else:
            if not joint:
                with torch.no_grad():
                    model.B.copy_(profile_head(g.detach(), residual[tr]))
            prediction = g @ model.B
            loss = ((prediction - residual[tr]).square() + prediction.square()).mean() + .01 * model.B.square().sum() / 20
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(params, 1.))
        optimizer.step()
        if not joint:
            with torch.no_grad():
                model.B.copy_(profile_head(model.features(delta, q[tr], ot[tr], history_kind), residual[tr]))
        val = validation_loss()
        row = [epoch, float(loss.detach()), val, gradient]
        if not np.isfinite(row).all():
            raise FloatingPointError('Nonfinite training state')
        trace.append(row)
        if val < best - 1e-6:
            best, selected, stale = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(best_state)
    log = dict(mode=mode, seed=int(seed), end=int(end), parameters=sum(p.numel() for p in model.parameters()),
               train_n=len(tr), validation_n=len(va), scale=scale,
               train_max_target=int(origins[tr[-1]] + 20), validation_first_origin=int(origins[va[0]]),
               validation_max_target=int(origins[va[-1]] + 20), selected_epoch=selected,
               epochs=len(trace), initial_validation=initial_validation, best_validation=best,
               history=trace, history_columns=['epoch', 'training_objective', 'validation_MSE', 'preclip_gradient_norm'])
    return model, scale, log


def predict(model, scale, price, origins, mode):
    base, context = reference(price)
    delta, q, ot, b = inputs(price, base, context, np.asarray(origins), scale)
    with torch.no_grad():
        return ((model.features(delta, q, ot, mode.split('_')[0]) @ model.B + b) * scale).numpy()
