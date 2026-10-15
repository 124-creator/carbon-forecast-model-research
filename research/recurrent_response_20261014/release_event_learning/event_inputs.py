"""Release-aligned arrays: known reports are messages, not daily observations."""
import numpy as np


def make_arrays(price, dates, source_dates, source_values, end, train_last, scale, stats=None):
    price, dates = np.asarray(price[:end]), np.asarray(dates[:end])
    count = int(np.searchsorted(source_dates, dates[-1], side='right'))
    sd = np.asarray(source_dates[:count])
    values = np.asarray(source_values[:count])
    if len(price) != end or not np.all(np.diff(dates) > 0) or not np.all(np.diff(sd) > 0):
        raise ValueError('Invalid chronological prefix')
    if values.shape != (count, 3) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Literal three-price vector required')
    logs = np.log(values)
    changes = np.diff(logs, axis=0, prepend=logs[:1])
    fit_count = int(np.searchsorted(sd, dates[train_last], side='right'))
    if stats is None:
        if fit_count < 3:
            raise ValueError('Insufficient released training reports')
        stats = dict(mean=logs[:fit_count].mean(0).tolist(),
                     level_scale=np.maximum(logs[:fit_count].std(0), 1e-6).tolist(),
                     change_scale=np.maximum(changes[:fit_count].std(0), 1e-6).tolist(),
                     fit_reports=fit_count, fit_date=int(dates[train_last]), scale=float(scale))
    levels = (logs - np.array(stats['mean'])) / np.array(stats['level_scale'])
    differences = changes / np.array(stats['change_scale'])
    gaps = np.diff(sd, prepend=sd[:1]) / 30.
    source = np.column_stack((levels, differences, gaps))
    neutral_source = np.zeros_like(source)
    neutral_source[:, -1] = gaps
    index = np.searchsorted(sd, dates[1:], side='right') - 1
    previous = np.r_[-1, index[:-1]]
    age = np.where(index >= 0, dates[1:] - sd[np.maximum(index, 0)], 0)
    last_gap = np.where(index >= 0, gaps[np.maximum(index, 0)], 0)
    clock = np.column_stack((index != previous, np.log1p(age) / np.log(31), index < 0, last_gap))
    arrays = dict(delta=np.diff(price) / scale, source=source, neutral_source=neutral_source,
                  source_index=index + 1, previous_index=previous + 1, clock=clock.astype(float))
    return arrays, stats
