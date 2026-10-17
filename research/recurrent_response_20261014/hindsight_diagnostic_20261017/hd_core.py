"""Observed-price feature representations for an explicitly hindsight search."""
import numpy as np


def feature_bank(price, dates, group):
    price, dates = np.asarray(price, dtype=float), np.asarray(dates)
    if (group not in ('price', 'dynamics', 'calendar') or price.ndim != 1 or
            len(price) != len(dates) or not len(price) or
            not np.isfinite(price).all() or np.any(price <= 0)):
        raise ValueError('Invalid observed feature support')
    delta = np.r_[0., np.diff(price)]
    columns = [delta]
    if group != 'price':
        mean5 = np.array([delta[max(1, t - 4):t + 1].mean() if t else 0. for t in range(len(price))])
        mean20 = np.array([delta[max(1, t - 19):t + 1].mean() if t else 0. for t in range(len(price))])
        std20 = np.array([delta[max(1, t - 19):t + 1].std() if t else 0. for t in range(len(price))])
        columns.extend([np.log(price), mean5, mean20, std20])
    if group == 'calendar':
        days = dates.astype('datetime64[D]').astype(np.int64)
        if np.any(np.diff(days) <= 0):
            raise ValueError('Dates are not strictly increasing')
        gap = np.r_[0, np.diff(days)]
        weekday = (days + 3) % 7
        columns.extend([np.log1p(gap), np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)])
    return np.stack(columns, axis=1)


def configuration_grid():
    from itertools import product
    configs = []
    for family, group, length, anchor, shrink in product(('gru', 'lstm'), ('price', 'dynamics', 'calendar'), (20, 60), ('rw', 'stat'), (0., 1.)):
        configs.append(dict(family=family, group=group, length=length, anchor=anchor, shrink=shrink,
                            hidden=16, lr=.003, weight_decay=.001, slice='factorial'))
    for family, hyper, shrink in product(('gru', 'lstm'), ((32, .003), (16, .001), (32, .001)), (0., 1.)):
        configs.append(dict(family=family, group='dynamics', length=60, anchor='stat', shrink=shrink,
                            hidden=hyper[0], lr=hyper[1], weight_decay=.001, slice='hyper_extension'))
    for group, length, anchor, alpha in product(('price', 'dynamics', 'calendar'), (20, 60), ('rw', 'stat'), (.01, .1, 1.)):
        configs.append(dict(family='ridge', group=group, length=length, anchor=anchor, shrink=0.,
                            alpha=alpha, slice='linear_control'))
    return [dict(id=f'c{i:03d}', **c) for i, c in enumerate(configs)]
