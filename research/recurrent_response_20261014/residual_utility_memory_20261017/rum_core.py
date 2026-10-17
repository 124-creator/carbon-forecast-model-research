"""Own bounded residual-memory primitives; no third-party model code."""
import numpy as np
import torch
from torch import nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fair_recurrent_tuning_20261016'))
from fair_core import RecurrentNet

MODES = ('gru', 'fixed', 'learned', 'utility')


def candidate_cache(price, ids, lookback=60, horizon=20, k=8):
    ids = np.asarray(ids, dtype=np.int64)
    if (ids.ndim != 1 or not len(ids) or ids.min() < lookback or
            np.any(np.diff(ids) <= 0) or ids.max() >= len(price) or k < 1):
        raise ValueError('Invalid query/key support')
    prefix = np.asarray(price[:int(ids.max()) + 1], dtype=float)
    if not np.isfinite(prefix).all():
        raise ValueError('Nonfinite observed price')
    windows = np.stack([np.diff(prefix[t - lookback:t + 1]) for t in ids])
    windows -= windows.mean(axis=1, keepdims=True)
    windows /= np.maximum(np.linalg.norm(windows, axis=1, keepdims=True), 1e-12)
    similarity = np.clip(windows @ windows.T, -1., 1.)
    allowed = ids[None, :] + horizon <= ids[:, None] - lookback
    order = np.argsort(-np.where(allowed, similarity, -np.inf), axis=1, kind='stable')
    take = min(k, len(ids))
    index = np.zeros((len(ids), k), dtype=np.int64)
    index[:, :take] = order[:, :take]
    valid = np.zeros_like(index, dtype=bool)
    valid[:, :take] = np.take_along_axis(allowed, order[:, :take], axis=1)
    chosen = np.take_along_axis(similarity, index, axis=1)
    chosen[~valid] = 0.
    age = np.log1p(np.maximum(ids[:, None] - ids[index], 0)) / np.log1p(500.)
    age[~valid] = 0.
    return dict(index=index, valid=valid, similarity=chosen, age=age)


def utility_teacher(error, memory, valid, horizon_variance, temperature=.25):
    with torch.no_grad():
        benefit = (error[:, None].square() - (error[:, None] - .25 * memory).square())
        benefit = benefit / horizon_variance.clamp_min(1e-8)[None, None]
        benefit = benefit.masked_fill(~valid[:, :, None], -1e9)
        logits = torch.cat((torch.zeros_like(benefit[:, :1]), benefit), dim=1)
        return torch.softmax(logits / temperature, dim=1)


class MemoryNet(RecurrentNet):
    def __init__(self, family):
        if family not in MODES:
            raise ValueError('Unknown memory family')
        super().__init__('gru_dense', 16)
        self.family = family
        self.scorer = nn.Sequential(nn.Linear(34, 16), nn.Tanh(), nn.Linear(16, 20))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.constant_(self.scorer[-1].bias, -4.)
        if family in ('gru', 'fixed'):
            self.scorer.requires_grad_(False)

    def memory_forward(self, features, cache, residual_values):
        plain = self.correction(features)
        index, valid = cache['index'], cache['valid']
        memory = residual_values - plain[index]
        if self.family == 'gru':
            scores = torch.full_like(memory, -1e9)
        elif self.family == 'fixed':
            scores = (2 * cache['similarity'][:, :, None] - 4).expand_as(memory)
        else:
            query = features[:, None, :8].expand(-1, index.shape[1], -1)
            key = features[index, :8]
            relation = torch.cat((query, key, query - key, (query - key).abs(),
                                  cache['similarity'][:, :, None], cache['age'][:, :, None]), dim=2)
            scores = self.scorer(relation)
        scores = scores.masked_fill(~valid[:, :, None], -1e9)
        logits = torch.cat((torch.zeros_like(scores[:, :1]), scores), dim=1)
        weights = torch.softmax(logits, dim=1)
        correction = plain + .25 * (weights[:, 1:] * memory).sum(dim=1)
        return correction, plain, memory, logits, weights
