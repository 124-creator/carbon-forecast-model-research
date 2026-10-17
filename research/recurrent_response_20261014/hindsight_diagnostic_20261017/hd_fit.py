"""Fresh ordinary ridge/GRU/LSTM fits; no diagnostic labels enter fitting."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import copy
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from hd_core import feature_bank
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fair_recurrent_tuning_20261016'))
from fair_core import configure, reference
from fair_data import support


class SequenceNet(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.encoder = (nn.GRU if config['family'] == 'gru' else nn.LSTM)(channels, config['hidden'], batch_first=True)
        self.readout = nn.Parameter(torch.zeros(config['hidden'] + 1, 20))

    def features(self, x):
        h = self.encoder(x)[0][:, -1]
        return torch.cat((h, torch.ones_like(h[:, :1])), dim=1)

    def forward(self, x):
        return self.features(x) @ self.readout


def windows(price, dates, origins, config):
    origins = np.asarray(origins, dtype=np.int64)
    if (origins.ndim != 1 or not len(origins) or origins.min() < config['length'] or
            np.any(np.diff(origins) <= 0) or origins.max() >= len(price)):
        raise ValueError('Invalid observed windows')
    stop = int(origins.max()) + 1
    bank = feature_bank(price[:stop], dates[:stop], config['group'])
    return bank[origins[:, None] - np.arange(config['length'] - 1, -1, -1)]


def base_forecast(price, config):
    return reference(price)[0] if config['anchor'] == 'stat' else np.zeros((len(price), 20))


def fit(price, dates, end, config, seed=17, max_epochs=60, deadline=None):
    configure()
    torch.manual_seed(seed)
    started = time.perf_counter()
    prefix, prefix_dates = np.array(price[:end], copy=True), np.array(dates[:end], copy=True)
    ids, train, validation, scale, target = support(prefix, end)
    raw = windows(prefix, prefix_dates, ids, config)
    mean, sd = raw[train].mean(axis=(0, 1)), np.maximum(raw[train].std(axis=(0, 1)), 1e-6)
    x = (raw - mean) / sd
    base = base_forecast(prefix, config)
    y = (target - base[ids]) / scale
    fitted = SimpleNamespace()
    fitted.config, fitted.mean, fitted.sd, fitted.scale = dict(config), mean, sd, scale
    trace, selected = [], 0
    if config['family'] == 'ridge':
        design = np.c_[x.reshape(len(ids), -1), np.ones(len(ids))]
        z = design[train]
        fitted.coef = z.T @ np.linalg.solve(z @ z.T + len(train) * config['alpha'] * np.eye(len(train)), y[train])
        fitted.net = None
        prediction = design @ fitted.coef
        initial = best = float(np.mean((prediction[validation] - y[validation]) ** 2))
        parameters = fitted.coef.size
        state_bytes = fitted.coef.tobytes()
    elif config['family'] in ('gru', 'lstm'):
        fitted.coef = None
        fitted.net = net = SequenceNet(config, x.shape[2])
        tensor, label = torch.as_tensor(x), torch.as_tensor(y)
        with torch.no_grad():
            z = net.features(tensor[train])
            coefficients = torch.linalg.solve((1 + config['shrink']) * z.T @ z / len(train) + .01 * torch.eye(z.shape[1]), z.T @ label[train] / len(train))
            net.readout.copy_(coefficients)
        optimizer = torch.optim.AdamW([dict(params=net.encoder.parameters(), weight_decay=config['weight_decay']),
                                       dict(params=[net.readout], weight_decay=0.)], lr=config['lr'])
        def validation_loss():
            with torch.no_grad():
                return float((net(tensor[validation]) - label[validation]).square().mean())
        best = initial = validation_loss()
        saved, stale = copy.deepcopy(net.state_dict()), 0
        for epoch in range(1, max_epochs + 1):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Original diagnostic deadline passed')
            optimizer.zero_grad(set_to_none=True)
            correction = net(tensor[train])
            train_mse = (correction - label[train]).square().mean()
            loss = train_mse + config['shrink'] * correction.square().mean() + .01 * net.readout.square().sum() / 20
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.))
            optimizer.step()
            value = validation_loss()
            row = [epoch, float(loss.detach()), float(train_mse.detach()), value, gradient]
            if not np.isfinite(row).all():
                raise FloatingPointError('Nonfinite fit; do not retry or fill')
            trace.append(row)
            if value < best - 1e-6:
                best, selected, stale = value, epoch, 0
                saved = copy.deepcopy(net.state_dict())
            else:
                stale += 1
            if stale >= 12:
                break
        net.load_state_dict(saved)
        with torch.no_grad():
            prediction = net(tensor).numpy()
        parameters = sum(p.numel() for p in net.parameters())
        state_bytes = b''.join(name.encode() + array.detach().numpy().tobytes() for name, array in sorted(net.state_dict().items()))
    else:
        raise ValueError('Unknown model family')
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite fitted predictions')
    original_validation = base[ids[validation]] + scale * prediction[validation]
    h = np.array([0, 4, 19])
    validation_primary = float(np.mean(np.mean((original_validation[:, h] - target[validation][:, h]) ** 2, axis=0) / np.mean(target[validation][:, h] ** 2, axis=0)))
    receipt = dict(config=dict(config), seed=seed, end=end, train_n=len(train), validation_n=len(validation),
                   pool_start=int(ids[0]), train_max_origin=int(ids[train[-1]]), train_max_target=int(ids[train[-1]] + 20),
                   validation_first_origin=int(ids[validation[0]]), validation_max_target=int(ids[-1] + 20),
                   feature_mean=mean.tolist(), feature_sd=sd.tolist(), target_scale=scale,
                   parameter_count=int(parameters), state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                   epochs=len(trace), selected_epoch=selected, initial_validation=initial, best_validation=best,
                   validation_primary=validation_primary, train_mse=float(np.mean((prediction[train] - y[train]) ** 2)),
                   seconds=time.perf_counter() - started, trace=trace,
                   trace_columns=['epoch', 'training_objective', 'prestep_train_MSE', 'poststep_validation_MSE', 'preclip_gradient'])
    return fitted, receipt


def predict(model, price, dates, origins):
    origins = np.asarray(origins, dtype=np.int64)
    raw = windows(price, dates, origins, model.config)
    x = (raw - model.mean) / model.sd
    if model.net is None:
        correction = np.c_[x.reshape(len(origins), -1), np.ones(len(origins))] @ model.coef
    else:
        with torch.no_grad():
            correction = model.net(torch.as_tensor(x)).numpy()
    base = base_forecast(np.asarray(price[:int(origins.max()) + 1]), model.config)
    return base[origins] + model.scale * correction
