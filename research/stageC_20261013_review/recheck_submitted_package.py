"""Independently score a locally supplied Stage C ZIP; never train/load weights.

Requires numpy. Default invocation checks aggregates without writing source rows.
Optional --out writes a JSON receipt to a caller-chosen local path.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

PREFIX = 'carbon_stageC_20261013/'
HORIZONS = (1, 5, 20)


def recheck(archive):
    with ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        if sum(i.file_size for i in z.infolist()) > 100 * 1024 * 1024:
            raise ValueError('Unexpected expanded archive size')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe member path')
        bad = z.testzip()
        if bad:
            raise ValueError('CRC failure: ' + bad)

        def data(rel):
            return z.read(PREFIX + rel)

        def records(rel):
            return list(csv.DictReader(io.StringIO(data(rel).decode('utf-8-sig'))))

        # Parse prices independently of submitted code, truncate before scoring.
        raw = data('raw_private/GCP.csv')
        source = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        points = [(datetime.strptime(r['date'], '%Y/%m/%d').strftime('%Y-%m-%d'),
                   float(r['price'])) for r in source]
        if len({d for d, _ in points}) != len(points) or points != sorted(points):
            raise ValueError('Non-unique or unsorted source dates')
        points = [(d, p) for d, p in points if d < '2023-01-01']
        dates = [d for d, _ in points]
        prices = np.asarray([p for _, p in points])
        if not np.isfinite(prices).all() or not (prices > 0).all():
            raise ValueError('Invalid source prices')
        n = len(prices)
        years = np.asarray([int(d[:4]) for d in dates])
        labels = np.full((n, 3), np.nan)
        for j, h in enumerate(HORIZONS):
            labels[:-h, j] = prices[h:] - prices[:-h]
        first = int(np.flatnonzero(years >= 2019)[0])
        scales = np.asarray([np.mean((prices[h:first] - prices[:first-h]) ** 2)
                             for h in HORIZONS])
        if not (scales > 0).all():
            raise ValueError('Invalid initial historical scales')
        idx = np.arange(n)
        masks = {year: (years == year) & (idx + 20 < n) &
                 (years[np.minimum(idx + 20, n - 1)] == year)
                 for year in (2019, 2020, 2021, 2022)}

        def score(pred, score_years):
            if pred.ndim != 3 or pred.shape[1:] != (n, 3):
                raise ValueError('Prediction shape mismatch')
            ratios, normalized = [], []
            for year in score_years:
                m = masks[year]
                if not m.any() or not np.isfinite(pred[:, m, :]).all():
                    raise ValueError('Missing score predictions')
                errors = np.mean((pred[:, m, :] - labels[m][None, :, :]) ** 2,
                                 axis=0)
                mse = np.mean(errors, axis=0)
                rw = np.mean(labels[m] ** 2, axis=0)
                if not (rw > 0).all():
                    raise ValueError('Zero RW score variance')
                ratios.extend(mse / rw)
                normalized.extend(mse / scales)
            return float(np.mean(ratios)), float(np.mean(normalized))

        prediction_cache = {}

        def prediction(tag):
            if tag not in prediction_cache:
                with np.load(io.BytesIO(data('outputs/predictions/' + tag + '.npz')),
                             allow_pickle=False) as arrays:
                    prediction_cache[tag] = arrays['delta'].copy()
            return prediction_cache[tag]

        registry = json.loads(data('outputs/configuration_registry.json'))
        old_tuning = {r['id']: r for r in records('outputs/all_tuning_configurations.csv')}
        by_family = defaultdict(list)
        tuning_max_difference = 0.0
        for entry in registry:
            tag = entry['id']
            relative, normalized = score(prediction(tag), (2019, 2020))
            tuning_max_difference = max(tuning_max_difference,
                abs(relative - float(old_tuning[tag]['tuning_relative_mse'])))
            by_family[entry['family']].append((relative, tag))
        selected = {kind: min(values)[1] for kind, values in by_family.items()}
        frozen = json.loads(data('outputs/selected_complete.json'))
        frozen_ids = {kind: item['id'] for kind, item in frozen.items()
                      if not kind.startswith('_')}
        if selected != frozen_ids:
            raise ValueError('Validation selection differs')
        model_tags = dict(selected)
        for kind in ('free', 'global'):
            matched = next(entry for entry in registry
                           if entry['family'] == kind and
                           entry['params'] == frozen['conditional']['params'])
            model_tags[kind + '_matched'] = matched['id']
        for kind in ('no_gap', 'fixed', 'no_drift'):
            model_tags['ablation_' + kind] = 'ablation_' + kind
        check_rows = records('outputs/check_summary.csv')
        check_max_difference = 0.0
        for row in check_rows:
            relative, normalized = score(prediction(model_tags[row['model']]), (2021, 2022))
            check_max_difference = max(check_max_difference,
                abs(relative - float(row['relative_mse'])),
                abs(normalized - float(row['NMSE'])))
        if max(tuning_max_difference, check_max_difference) > 1e-12:
            raise ValueError('Score difference exceeds tolerance')
        return {
            'archive_sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            'source_GCP_sha256': hashlib.sha256(raw).hexdigest(),
            'source_rows': len(source), 'pre2023_rows': n,
            'scoring_origins': {str(k): int(v.sum()) for k, v in masks.items()},
            'tuning_configurations': len(registry), 'check_summary_rows': len(check_rows),
            'tuning_max_abs_difference': tuning_max_difference,
            'check_max_abs_difference': check_max_difference,
            'validation_selections_match': True,
            'model_retraining_performed': False, 'pickle_loading_performed': False,
            'source_rows_written': False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = recheck(args.archive)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
