"""Strong supervised GRU baselines for the fixed target-label-budget experiment.

These are standard local/pooled predictors, not novel models. Both may use the
target support AND meta labels (80/320 supervised origins), while validation is
only for early stopping. They never use test labels. The caller owns data reads
and saving results; this module accepts in-memory cases from data.make_case.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from torch import nn


DEFAULT_CONFIG = {
    'lr': 0.003,
    'weight_decay': 1e-4,
    'max_steps': 600,
    'batch_size': 64,
    'validation_every': 10,
    'patience': 10,
    'min_delta': 1e-6,
    'sampler_seed': 17018,
}
HORIZONS = np.array([0, 4, 19], dtype=np.int64)


class SupervisedGRU(nn.Module):
    """One GRU16 plus an independent zero-initialized Linear16->20 per market."""
    def __init__(self, markets):
        super().__init__()
        self.market_names = tuple(markets)
        if len(set(self.market_names)) != len(self.market_names):
            raise ValueError('Duplicate market name')
        self.market_index = {name: i for i, name in enumerate(self.market_names)}
        self.encoder = nn.GRU(3, 16, num_layers=1, batch_first=True,
                              device='cpu', dtype=torch.float64)
        self.heads = nn.ModuleList([nn.Linear(16, 20, device='cpu', dtype=torch.float64)
                                   for _ in self.market_names])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, market):
        _, final = self.encoder(x)
        return self.heads[self.market_index[market]](final[-1])


def _config(config):
    config = {} if config is None else dict(config)
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f'Unknown baseline config keys: {sorted(unknown)}')
    out = dict(DEFAULT_CONFIG)
    out.update(config)
    for name in ('max_steps', 'batch_size', 'validation_every', 'patience'):
        if not isinstance(out[name], int) or out[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if out['lr'] <= 0 or out['weight_decay'] < 0 or out['min_delta'] < 0:
        raise ValueError('Invalid optimizer or early-stopping parameter')
    return out


def _features(batch):
    x = np.asarray(batch['x'], dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (20, 3) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty x[N,20,3]')
    return np.ascontiguousarray(x)


def _supervised(batch):
    x = _features(batch)
    y = np.asarray(batch['y'], dtype=np.float64)
    if y.shape != (len(x), 20) or not np.isfinite(y).all():
        raise ValueError('Expected finite y[N,20] in scaled-increment units')
    return torch.from_numpy(x.copy()), torch.from_numpy(np.ascontiguousarray(y).copy())


def _state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8') + b'\x00')
        digest.update(str(array.dtype).encode() + b'\x00')
        digest.update(str(array.shape).encode() + b'\x00')
        digest.update(array.tobytes())
    return digest.hexdigest()


def _predict(model, x, market):
    model.eval()
    with torch.no_grad():
        out = model(x, market).detach().cpu().numpy().copy()
    if not np.isfinite(out).all():
        raise FloatingPointError('Nonfinite baseline prediction')
    return out


def _validation_inputs(batch):
    x = torch.from_numpy(_features(batch).copy())
    y = np.asarray(batch['y'], dtype=np.float64)
    sigma = np.asarray(batch['sigma'], dtype=np.float64)
    if y.shape != (len(x), 20) or sigma.shape != y.shape or not np.isfinite(y).all():
        raise ValueError('Bad validation shape')
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('Validation sigma must be finite and positive')
    delta = np.asarray(batch['delta'], dtype=np.float64) if 'delta' in batch else y * sigma
    if delta.shape != y.shape or not np.isfinite(delta).all():
        raise ValueError('Bad validation delta')
    if not np.allclose(delta, y * sigma, atol=1e-10, rtol=1e-10):
        raise ValueError('Validation y/sigma do not reconstruct its raw delta')
    denominator = np.mean(delta ** 2, axis=0)
    if np.any(denominator[HORIZONS] <= 0):
        raise ValueError('Validation RW denominator is zero; no silent floor')
    return x, sigma, delta, denominator


def _validation(model, x, sigma, delta, denominator, market):
    predictions = _predict(model, x, market)
    mse = np.mean((predictions * sigma - delta) ** 2, axis=0)
    ratios = mse[HORIZONS] / denominator[HORIZONS]
    score = float(ratios.mean())
    if not np.isfinite(score):
        raise FloatingPointError('Nonfinite validation score')
    return predictions, {
        'score': score,
        'raw_mse_h1_h5_h20': mse[HORIZONS].tolist(),
        'rw_mse_h1_h5_h20': denominator[HORIZONS].tolist(),
        'relative_mse_h1_h5_h20': ratios.tolist(),
    }


def _source_seed(base, market):
    key = f'{int(base)}|baseline_source|{market}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')


def fit_baseline(case, name, seed=17, config=None):
    """Fit one standard baseline; return scaled test predictions and selected state.

    Training is full-20-dimensional MSE of supplied y=delta/sigma. Model/epoch
    selection uses raw-price MSE divided by the same validation RW MSE, equally
    over h1/h5/h20. Test x is accessed once, after loading the selected state;
    test y/delta/base/date are never accessed.

    Sampling uses independent NumPy streams for target and for each source.
    Source schedules therefore do not change merely because the target label
    budget changes. Within-batch sampling is without replacement when n>=64.
    """
    if name not in ('local_gru', 'pooled_gru'):
        raise ValueError('Unknown baseline family')
    cfg = _config(config)
    started = time.perf_counter()
    torch.set_num_threads(1)
    target = str(case['target'])
    support_x, support_y = _supervised(case['target_support'])
    meta_x, meta_y = _supervised(case['target_meta'])
    target_x = torch.cat((support_x, meta_x), dim=0)
    target_y = torch.cat((support_y, meta_y), dim=0)
    sources = case['sources'] if name == 'pooled_gru' else {}
    if name == 'pooled_gru' and not sources:
        raise ValueError('Pooled baseline requires at least one source')
    if target in sources:
        raise ValueError('Target cannot also appear in source pool')
    source_names = tuple(sorted(sources))
    datasets = {target: (target_x, target_y)}
    datasets.update({market: _supervised(sources[market]) for market in source_names})
    market_names = (target,) + source_names

    # Keep initialization deterministic without consuming the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = SupervisedGRU(market_names).to(device='cpu', dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']),
                                 weight_decay=float(cfg['weight_decay']), foreach=False)
    initial_state = _state(model)
    initial_hash = state_sha256(initial_state)
    val_x, val_sigma, val_delta, val_denominator = _validation_inputs(case['target_val'])
    best_val_predictions, val = _validation(model, val_x, val_sigma, val_delta, val_denominator, target)
    best_state, best_score, best_step = initial_state, val['score'], 0
    initial_predictions_are_rw = bool(np.array_equal(best_val_predictions, np.zeros_like(best_val_predictions)))
    if not initial_predictions_are_rw:
        raise AssertionError('Zero-head step 0 must equal RW')
    traces = [{'step': 0, 'validation': val, 'selected': True, 'training_loss': None}]
    bad_checks, validation_calls, steps = 0, 1, 0
    stop_reason = 'max_steps'

    seeds = {target: int(cfg['sampler_seed'])}
    seeds.update({market: _source_seed(cfg['sampler_seed'], market) for market in source_names})
    samplers = {market: np.random.default_rng(seeds[market]) for market in market_names}
    sampler_digests = {market: hashlib.sha256() for market in market_names}
    weights = {target: 1.0} if name == 'local_gru' else {
        target: 0.5, **{market: 0.5 / len(source_names) for market in source_names}}
    for step in range(1, cfg['max_steps'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, component_values = [], {}
        for market in market_names:
            x, y = datasets[market]
            ids = samplers[market].choice(len(x), size=cfg['batch_size'], replace=len(x) < cfg['batch_size'])
            sampler_digests[market].update(np.asarray(ids, dtype='<i8').tobytes())
            ids_t = torch.from_numpy(np.asarray(ids, dtype=np.int64))
            prediction = model(x[ids_t], market)
            component = torch.mean((prediction - y[ids_t]) ** 2)
            component_values[market] = float(component.detach())
            losses.append(weights[market] * component)
        loss = torch.stack(losses).sum()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite baseline training loss')
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError('Nonfinite baseline gradient')
        optimizer.step()
        steps = step
        row = {'step': step, 'training_loss': float(loss.detach()),
               'per_market_training_mse': component_values, 'validation': None, 'selected': False}
        if step % cfg['validation_every'] == 0 or step == cfg['max_steps']:
            predictions, val = _validation(model, val_x, val_sigma, val_delta, val_denominator, target)
            validation_calls += 1
            row['validation'] = val
            if val['score'] < best_score - cfg['min_delta']:
                best_score, best_step = val['score'], step
                best_state, best_val_predictions = _state(model), predictions.copy()
                bad_checks = 0
                row['selected'] = True
            else:
                bad_checks += 1
            row['consecutive_nonimproving_validation_checks'] = bad_checks
        traces.append(row)
        if bad_checks >= cfg['patience']:
            stop_reason = 'early_stop_patience'
            break

    last_state_hash = state_sha256(_state(model))
    model.load_state_dict(best_state)
    # First and only test forward; no test labels or metrics are accessed here.
    test_x = torch.from_numpy(_features(case['target_test']).copy())
    test_predictions = _predict(model, test_x, target)
    receipt = {
        'family': name, 'seed': int(seed), 'target': target, 'config': cfg,
        'architecture': 'GRU(input3,hidden16,one_layer) plus independent Linear16_to20 per market',
        'novelty': 'standard strong supervised baseline; no innovation claim',
        'dtype': 'torch.float64', 'device': 'cpu', 'torch_threads': 1,
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'target_supervision': 'support plus meta, legally inside the retained target budget; validation excluded',
        'target_support_count': len(support_x), 'target_meta_count': len(meta_x),
        'target_training_count': len(target_x), 'validation_count': len(val_x),
        'source_counts': {market: len(datasets[market][0]) for market in source_names},
        'source_cutoff': case.get('source_cutoff'), 'source_pool': 'entire fixed caller-approved source pool, no later crop',
        'training_loss': 'full20 mean squared scaled increment error',
        'market_loss_weights': weights,
        'selection_score': 'equal h1,h5,h20 raw-price MSE divided by same-cell validation RW MSE',
        'zero_head_initialization': True, 'initial_predictions_equal_rw': initial_predictions_are_rw,
        'initial_state_sha256': initial_hash,
        'initial_encoder_hash': state_sha256({k:v for k,v in initial_state.items() if k.startswith('encoder.')}),
        'last_state_sha256': last_state_hash,
        'selected_state_sha256': state_sha256(best_state),
        'steps_completed': steps, 'validation_calls': validation_calls,
        'best_step': best_step, 'best_validation_score': best_score,
        'stop_reason': stop_reason, 'test_forward_calls': 1, 'test_labels_accessed': False,
        'sampler': 'independent target/source NumPy streams; no replacement within a batch when enough origins',
        'sampler_seeds': seeds,
        'sampled_indices_sha256': {market: sampler_digests[market].hexdigest() for market in market_names},
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'elapsed_seconds': time.perf_counter() - started,
    }
    return {'predictions': test_predictions, 'val_predictions': best_val_predictions,
            'traces': traces, 'state': best_state, 'receipt': receipt}
