"""Exact scoring of the stated finite predictive measure, not an iid ensemble."""
import numpy as np


def crps_numpy(atoms, probs, y):
    a, p, y = np.asarray(atoms), np.asarray(probs), np.asarray(y)
    order = np.argsort(a, axis=-1, kind='stable')
    a, p = np.take_along_axis(a, order, -1), np.take_along_axis(p, order, -1)
    before = np.cumsum(p, axis=-1)-p
    return np.sum(p*np.abs(a-y[..., None]), axis=-1)-np.sum(p*a*(2*before+p-1), axis=-1)


def crps_torch(atoms, probs, y):
    import torch
    a, ix = torch.sort(atoms, dim=-1)
    p = torch.gather(probs, -1, ix)
    before = torch.cumsum(p, -1)-p
    return (p*torch.abs(a-y.unsqueeze(-1))).sum(-1)-(p*a*(2*before+p-1)).sum(-1)


def quantile(a, p, tau):
    order = np.argsort(a, axis=-1, kind='stable')
    a, p = np.take_along_axis(a, order, -1), np.take_along_axis(p, order, -1)
    index = np.argmax(np.cumsum(p, axis=-1) >= tau, axis=-1)
    return np.take_along_axis(a, index[..., None], -1)[..., 0]


def evaluate(a, p, batch):
    a, p, y = np.asarray(a), np.asarray(p), np.asarray(batch['delta'])
    if a.ndim != 3 or p.ndim != 3 or y.ndim != 2 or not len(y) or a.shape[-1] == 0:
        raise ValueError('Predictive measures require shape (origins,horizons,atoms)')
    if a.shape != p.shape or a.shape[:2] != y.shape or not np.isfinite(a).all() or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid predictive measure')
    if np.any(p < 0) or np.max(np.abs(p.sum(-1)-1)) > 1e-10:
        raise ValueError('Invalid probabilities')
    loss = crps_numpy(a, p, y)
    mean = np.sum(a*p, axis=-1)
    rms = np.sqrt(np.mean(y*y, axis=0))
    median = quantile(a, p, .5)
    out = dict(n=len(y), crps=loss.mean(0).tolist(),
               mean_mse=np.mean((mean-y)**2, 0).tolist(), rw_mse=np.mean(y*y, 0).tolist(),
               mean_mse_ratio=(np.mean((mean-y)**2, 0)/np.mean(y*y, 0)).tolist(),
               median_mae=np.mean(abs(median-y), 0).tolist(),
               max_abs_predictive_change_mean=float(np.max(abs(mean))),
               minimum_crps=float(loss.min()), probability_sum_error=float(np.max(abs(p.sum(-1)-1))))
    for level in (.5, .8, .9):
        lo, hi = quantile(a, p, (1-level)/2), quantile(a, p, 1-(1-level)/2)
        key = str(int(round(level*100)))
        out['coverage_'+key] = np.mean((y>=lo)&(y<=hi), 0).tolist()
        out['width_'+key] = np.mean(hi-lo, 0).tolist()
        out['width_over_rw_rms_'+key] = (np.mean(hi-lo, 0)/rms).tolist()
    return out, loss
