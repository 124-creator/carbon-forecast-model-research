"""Bounded approved-prefix access and strictly mature chronological pools."""
import csv
import hashlib
import io
from pathlib import Path
import numpy as np

PREFIX_SHA = '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'

def load_prefix(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[3] / 'data/parsed/figshare_28682348/guangzhou_reported.csv'
    with Path(path).open('rb', buffering=0) as stream:
        raw = stream.read(29204)
    if len(raw) != 29204 or hashlib.sha256(raw).hexdigest() != PREFIX_SHA:
        raise ValueError('Approved prefix identity differs; do not read a suffix')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    dates = np.array([row['date'] for row in rows])
    price = np.array([float(row['author_reported_price']) for row in rows])
    if len(rows) != 950 or dates[-1] != '2021-12-31' or not np.all(dates[1:] > dates[:-1]):
        raise ValueError('Approved date support differs')
    if not np.isfinite(price).all() or np.any(price <= 0):
        raise ValueError('Invalid original prices')
    return price, dates, dict(records=len(rows), bytes=len(raw), last_date=str(dates[-1]),
                              prefix_sha256=PREFIX_SHA, next_byte_requested=False)

def support(price, end):
    if end > len(price) or end < 221:
        raise ValueError('Insufficient causal fit support')
    origins = np.arange(max(60, end - 20 - 480), end - 20)
    split = len(origins) - 60
    train = np.arange(split - 20)
    validation = np.arange(split, len(origins))
    if len(train) < 60 or origins[train[-1]] + 20 >= origins[validation[0]] or origins[-1] + 20 >= end:
        raise ValueError('Maturity or validation purge violation')
    windows = np.stack([np.diff(price[t - 60:t + 1]) for t in origins[train]])
    scale = max(float(windows.std()), 1e-6)
    target = np.stack([price[t + 1:t + 21] - price[t] for t in origins])
    return origins, train, validation, scale, target
