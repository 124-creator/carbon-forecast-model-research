"""Fresh chronological training with explicit parameter-vintage boundaries."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import copy
import time
import numpy as np
import torch
from torch import nn
from innovation_model import InnovationNet, basis
from reference_support import configure, reference, support

MODES = ('innovation_shift', 'raw_shift', 'innovation_diagonal', 'constant_shift',
         'gru_profile', 'gru_rank_window', 'lstm_profile')


def ridge(features, residual):
    n, d = features.shape
    eye = torch.eye(d, dtype=features.dtype, device=features.device)
    return torch.linalg.solve(2 * features.T @ features / n + .01 * eye, features.T @ residual / n)


class OrdinaryNet(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.encoder = (nn.LSTM if mode == 'lstm_profile' else nn.GRU)(1, 16, batch_first=True)
        self.U = nn.Linear(16, 8, bias=False)
        self.V = nn.Linear(3, 8, bias=False)
        self.c = nn.Parameter(torch.zeros(8))
        if mode == 'gru_rank_window':
            self.B = nn.Parameter(torch.zeros(9, 6))
            self.F = nn.Parameter(basis().T.clone())
        else:
            self.register_buffer('B', torch.zeros(9, 20))

    def features(self, delta, context, origins):
        if self.mode == 'gru_rank_window':
            windows = torch.stack([delta[int(t) - 60:int(t)] for t in origins])
            hidden = self.encoder(windows[:, :, None])[0][:, -1]
            neutral = self.encoder(torch.zeros_like(windows[:1])[:, :, None])[0][0, -1]
            hidden = hidden - neutral
        else:
            observed = self.encoder(delta[None, :, None])[0][0]
            neutral = self.encoder(torch.zeros_like(delta)[None, :, None])[0][0]
            hidden = torch.cat((torch.zeros_like(observed[:1]), observed - neutral))[origins]
        f = torch.tanh(self.U(hidden) + self.V(context[origins]) + self.c) - torch.tanh(self.c)
        return torch.cat((f, torch.ones_like(f[:, :1])), dim=1)

    def forward(self, delta, context, base, origins):
        correction = self.features(delta, context, origins) @ self.B
        if self.mode == 'gru_rank_window':
            correction = correction @ self.F
        return base[origins] + correction


def tensors(price, scale):
    base, context = reference(price)
    context = context.copy()
    context[:, 0] = np.tanh(context[:, 0] / scale)
    return (torch.as_tensor(np.diff(price) / scale), torch.as_tensor(context), torch.as_tensor(base / scale))


def output(model, args, origins):
    if isinstance(model, InnovationNet):
        return model(*args)[origins]
    return model(*args, origins)


def fit(price, end, mode, seed, max_epochs=60, deadline=None):
    if mode not in MODES:
        raise ValueError(mode)
    configure()
    torch.manual_seed(int(seed))
    prefix = np.array(price[:end], copy=True)
    origins, tr, va, scale, target = support(prefix, end)
    args = tensors(prefix, scale)
    indices = torch.as_tensor(origins, dtype=torch.long)
    tr_ids, va_ids = indices[tr], indices[va]
    truth = torch.as_tensor(target / scale)
    model = InnovationNet(mode) if mode in MODES[:4] else OrdinaryNet(mode)
    is_profile = mode.endswith('_profile')
    is_rank = mode == 'gru_rank_window'

    def install(initial=False):
        if not is_profile and not (initial and is_rank):
            return
        with torch.no_grad():
            features = model.features(args[0], args[1], tr_ids)
            fitted = ridge(features, truth[tr] - args[2][tr_ids])
            model.B.copy_(fitted @ model.F.T if is_rank else fitted)

    def validation():
        with torch.no_grad():
            return float((output(model, args, va_ids) - truth[va]).square().mean())

    install(initial=True)
    initial = best = validation()
    best_state = copy.deepcopy(model.state_dict())
    chosen = stale = 0
    trace = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    for epoch in range(1, max_epochs + 1):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError('Frozen market budget exhausted')
        optimizer.zero_grad(set_to_none=True)
        prediction = output(model, args, tr_ids)
        correction = prediction - args[2][tr_ids]
        loss = ((prediction - truth[tr]).square() + correction.square()).mean()
        if is_profile:
            loss = loss + .01 * model.B.square().sum() / 20
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
    log = dict(mode=mode, seed=int(seed), end=int(end), train_n=len(tr), validation_n=len(va),
               scale=scale, train_max_target=int(origins[tr[-1]] + 20),
               validation_first_origin=int(origins[va[0]]), validation_max_target=int(origins[va[-1]] + 20),
               parameters=sum(p.numel() for p in model.parameters()),
               analytic_head_values=int(model.B.numel()) if is_profile else 0,
               initial_validation=initial, best_validation=best, selected_epoch=chosen,
               epochs=len(trace), history_columns=['epoch', 'objective', 'validation_MSE', 'gradient_norm_preclip'], history=trace)
    return model, scale, log


def predict(model, scale, price, origins, mode):
    del mode
    origins = np.asarray(origins, dtype=np.int64)
    prefix = np.array(price[:int(origins.max()) + 1], copy=True)
    args = tensors(prefix, scale)
    with torch.no_grad():
        return (output(model, args, torch.as_tensor(origins, dtype=torch.long)) * scale).numpy()
