"""Producer metrics; verifier must not import this module."""
import numpy as np

def span_score(prediction, truth):
    if prediction.ndim != 3 or truth.ndim != 2 or prediction.shape[1:] != truth.shape or truth.shape[1] != 20 or not len(truth):
        raise ValueError('Invalid prediction support')
    ratios, ensembles, cells = [], [], []
    for h in (1, 5, 20):
        actual = truth[:, h - 1]
        forecast = prediction[:, :, h - 1]
        denominator = float(np.mean(actual ** 2))
        if denominator <= 0 or not np.isfinite(forecast).all() or not np.isfinite(actual).all():
            raise ValueError('Undefined cell or nonfinite forecasts')
        values = np.mean((forecast - actual) ** 2, axis=1) / denominator
        ratios.append(values)
        ensembles.append(float(np.mean((forecast.mean(0) - actual) ** 2) / denominator))
        cells.append(dict(horizon=h, count=len(actual), ratio=float(values.mean())))
    seeds = np.mean(ratios, axis=0)
    return dict(score=float(seeds.mean()), seeds=seeds.tolist(), ensemble=float(np.mean(ensembles)), cells=cells)

def score(prediction, truth, years):
    seeds, ensembles, cells, annual = [], [], [], {}
    for year in ('2020', '2021'):
        subset = years == year
        if not subset.any():
            raise ValueError('Missing required year')
        result = span_score(prediction[:, subset], truth[subset])
        seeds.append(result['seeds'])
        ensembles.append(result['ensemble'])
        annual[year] = result['score']
        cells.extend(dict(cell, year=year) for cell in result['cells'])
    seed_scores = np.mean(seeds, axis=0)
    return dict(score=float(seed_scores.mean()), seeds=seed_scores.tolist(), ensemble=float(np.mean(ensembles)),
                cells=cells, annual=annual)
