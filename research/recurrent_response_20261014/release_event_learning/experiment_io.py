"""Bounded raw-prefix access and transparent, equal-cell forecast scores."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX_SHA = '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_prefix():
    reader_path = ROOT / 'research/torch_segmented_backbone_20261014/prefix_loader.py'
    spec = importlib.util.spec_from_file_location('_approved_carbon_prefix_reader', reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    rows, receipt = reader.read_prefix(ROOT / 'data/parsed/figshare_28682348/guangzhou_reported.csv', 950, '2021-12-31')
    if receipt['prefix_sha256'] != PREFIX_SHA:
        raise ValueError('Approved prefix identity changed')
    price = np.array([float(row['author_reported_price']) for row in rows])
    dates = np.array([row['date'] for row in rows])
    if not np.isfinite(price).all() or not np.all(dates[1:] > dates[:-1]):
        raise ValueError('Invalid raw prefix')
    return price, dates, receipt


def summarize(prediction, truth, years):
    ratios, ensembles, cells, annual = [], [], [], {}
    for year in ('2020', '2021'):
        mask = years == year
        year_ratios = []
        if not mask.any():
            raise ValueError('Missing required year')
        for horizon in (1, 5, 20):
            target = truth[mask, horizon - 1]
            denominator = np.mean(target ** 2)
            if denominator <= 0:
                raise ValueError('Undefined random-walk comparison')
            errors = prediction[:, mask, horizon - 1] - target
            seed_ratios = np.mean(errors ** 2, axis=1) / denominator
            ratios.append(seed_ratios)
            year_ratios.append(float(seed_ratios.mean()))
            ensembles.append(float(np.mean((prediction[:, mask, horizon - 1].mean(0) - target) ** 2) / denominator))
            cells.append(dict(year=year, horizon=horizon, count=int(mask.sum()), ratio=float(seed_ratios.mean())))
        annual[year] = float(np.mean(year_ratios))
    seeds = np.mean(ratios, axis=0)
    return dict(score=float(seeds.mean()), seeds=seeds.tolist(), ensemble=float(np.mean(ensembles)), cells=cells, annual=annual)
