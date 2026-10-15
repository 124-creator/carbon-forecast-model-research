"""Independent M/N/O score replay. Imports no producer model, runner or scorer."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent


def verify(artifact_root, repo_root):
    targets = json.loads((HERE / 'REMOTE_TARGETS.json').read_text(encoding='utf-8'))
    receipts = json.loads((HERE / 'SOURCE_RECEIPTS.json').read_text(encoding='utf-8'))
    receipt_index = {(r['commit'], r['path']): r for r in receipts}
    with (repo_root / 'data/parsed/figshare_28682348/guangzhou_reported.csv').open('rb', buffering=0) as stream:
        raw = stream.read(29204)
    if hashlib.sha256(raw).hexdigest() != '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070':
        raise ValueError('Approved price-prefix identity changed')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(rows) != 950 or rows[-1]['date'] != '2021-12-31':
        raise ValueError('Wrong source prefix')
    price = np.array([float(r['author_reported_price']) for r in rows])
    dates = np.array([r['date'] for r in rows])
    origins = np.arange(459, 930)
    truth_all = np.array([[price[o + h] - price[o] for h in range(1, 21)] for o in origins])
    year_all = dates[origins].astype('U4')
    eligible = year_all == dates[origins + 20].astype('U4')
    truth, years = truth_all[eligible], year_all[eligible]
    if len(truth) != 451 or int(sum(years == '2020')) != 224 or int(sum(years == '2021')) != 227:
        raise ValueError('Wrong score population')
    differences = []
    verified_blobs = set()
    output = {}
    total = 0

    def checked_file(fragment, relative):
        entry = targets[fragment]
        key = (entry['commit'], entry['prefix'] + relative)
        receipt = receipt_index[key]
        path = artifact_root / fragment / relative
        payload = path.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        if sha != receipt['sha256'] or blob != receipt['blob']:
            raise ValueError('Pinned file changed: ' + str(key))
        verified_blobs.add(key)
        if path.suffix == '.npz':
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len({x.filename for x in members}) != len(members) or sum(x.file_size for x in members) > 20000000:
                    raise ValueError('Invalid or excessive numeric archive')
        return path

    def metric(pred):
        if pred.ndim != 3 or pred.shape[1:] != (451, 20) or pred.shape[0] not in (1, 4):
            raise ValueError('Unexpected prediction shape')
        if not np.isfinite(pred[:, :, [0, 4, 19]]).all():
            raise ValueError('Missing evaluated horizon')
        cells, seed_cells, ensemble_cells = [], [], []
        for year in ['2020', '2021']:
            mask = years == year
            for h in [1, 5, 20]:
                observed = truth[mask, h - 1]
                error = pred[:, mask, h - 1] - observed
                denominator = float(observed @ observed)
                ratios = np.sum(error * error, axis=1) / denominator
                seed_cells.append(ratios)
                ensemble_error = pred[:, mask, h - 1].mean(axis=0) - observed
                ensemble_cells.append(float(ensemble_error @ ensemble_error / denominator))
                cells.append(dict(year=year, horizon=h, n=int(sum(mask)), ratio=float(ratios.mean()),
                                  RMSE_seed_loss=float(np.sqrt(np.sum(error * error) / error.size)),
                                  MAE_seed_mean=float(np.sum(np.abs(error)) / error.size)))
        seed_scores = np.array(seed_cells).mean(axis=0)
        return dict(score=float(seed_scores.mean()), ensemble=float(np.mean(ensemble_cells)),
                    seed_scores=seed_scores.tolist(), cells=cells)

    def compare(actual, stored):
        for key in ['score', 'ensemble']:
            differences.append(abs(actual[key] - stored[key]))
        if len(actual['seed_scores']) != len(stored['seed_scores']):
            raise ValueError('Seed-count disagreement')
        differences.extend(abs(a - b) for a, b in zip(actual['seed_scores'], stored['seed_scores']))
        if len(stored['cells']) != 6:
            raise ValueError('Wrong published cell count')
        for actual_cell, stored_cell in zip(actual['cells'], stored['cells']):
            if any(actual_cell[k] != stored_cell[k] for k in ['year', 'horizon', 'n']):
                raise ValueError('Cell order or support disagreement')
            differences.extend(abs(actual_cell[k] - stored_cell[k]) for k in ['ratio', 'RMSE_seed_loss', 'MAE_seed_mean'])

    for fragment, entry in targets.items():
        directory = artifact_root / fragment / 'results_remote'
        expected = {Path(name).stem for name in entry['predictions']}
        if len(expected) != 29 or {p.stem for p in (directory / 'predictions').glob('*.npz')} != expected:
            raise ValueError('Missing, extra or duplicated configuration')
        with np.load(checked_file(fragment, 'results_remote/EVALUATION.npz'), allow_pickle=False) as evaluation:
            for key, actual in [('origins', origins), ('truth', truth_all), ('years', year_all), ('eligible', eligible)]:
                if not np.array_equal(evaluation[key], actual):
                    raise ValueError('Original-label reconstruction mismatch: ' + fragment + '/' + key)
        stored = json.loads(checked_file(fragment, 'results_remote/ALL_MODEL_METRICS.json').read_text(encoding='utf-8'))
        if set(stored) != expected:
            raise ValueError('Published metric configuration set mismatch')
        all_metrics = {}
        for name in sorted(expected):
            with np.load(checked_file(fragment, 'results_remote/predictions/' + name + '.npz'), allow_pickle=False) as numeric:
                array = numeric['prediction']
                if array.dtype.kind != 'f' or array.shape[1:] != (471, 20):
                    raise ValueError('Invalid prediction array')
                all_metrics[name] = metric(array[:, eligible])
            compare(all_metrics[name], stored[name])
            total += 1
        candidate_name = {'renewal-response': 'RENEWAL', 'adaptive-horizon': 'MESH', 'forward-gain': 'GAIN'}[fragment]
        candidate = all_metrics[candidate_name]
        pool = [k for k in all_metrics if k != candidate_name]
        best = min(pool, key=lambda k: all_metrics[k]['score'])
        best_ensemble = min(pool, key=lambda k: all_metrics[k]['ensemble'])
        annual = {y:float(np.mean([c['ratio'] for c in candidate['cells'] if c['year'] == y])) for y in ['2020', '2021']}
        controls = {'renewal-response':['MARKOV','CLOCK','RANK4','RANK6','FIXED6','DIRECT'],
                    'adaptive-horizon':['FIXED6','GLOBAL','TILT','RANK4','RANK6','DIRECT'],
                    'forward-gain':['FWD_FIXED','IN_SAMPLE','VARPRO_FIXED','FROZEN_GAIN','FIXED6','RANK6']}[fragment]
        gains = {k:1 - candidate['score'] / all_metrics[k]['score'] for k in set(controls + ['RW', best])}
        wins = {k:int(sum(a < b for a,b in zip(candidate['seed_scores'], all_metrics[k]['seed_scores']))) for k in controls}
        if fragment == 'renewal-response':
            mechanism = all(gains[k] >= .01 for k in ['MARKOV','CLOCK','RANK6','FIXED6']) and wins['MARKOV'] >= 3 and wins['CLOCK'] >= 3
        elif fragment == 'adaptive-horizon':
            mechanism = all(gains[k] >= .01 for k in ['FIXED6','GLOBAL','TILT']) and wins['FIXED6'] >= 3 and wins['TILT'] >= 3
        else:
            mechanism = all(gains[k] >= .01 for k in ['FWD_FIXED','IN_SAMPLE','VARPRO_FIXED','FROZEN_GAIN']) and wins['FWD_FIXED'] >= 3 and wins['IN_SAMPLE'] >= 3
        best_gate = gains[best] > 0 if fragment == 'forward-gain' else gains[best] >= .01
        accuracy = gains['RW'] >= .02 and best_gate and candidate['ensemble'] < all_metrics[best_ensemble]['ensemble'] and max(annual.values()) <= 1.05
        summary = json.loads(checked_file(fragment, 'results_remote/RESULTS.json').read_text(encoding='utf-8'))
        compare(candidate, summary['candidate'])
        if bool(mechanism) != summary['mechanism_pass'] or bool(accuracy) != summary['accuracy_pass']:
            raise ValueError('Independent gate decision differs')
        for k, value in summary['gains'].items():
            differences.append(abs(value - gains[k]))
        if wins != summary['seed_wins'] or summary['best_non_candidate'] != best or summary['best_ensemble_non_candidate'] != best_ensemble:
            raise ValueError('Ranking or paired-seed decision differs')
        output[fragment] = dict(commit=entry['commit'], candidate=candidate_name, configurations=len(expected),
                               mechanism_pass=bool(mechanism), accuracy_pass=bool(accuracy), annual=annual,
                               best_non_candidate=best, best_ensemble_non_candidate=best_ensemble,
                               gains=gains, seed_wins=wins, scores=all_metrics)
    maximum = max(differences)
    if total != 87 or not np.isfinite(differences).all() or maximum > 1e-10:
        raise ValueError('Incomplete or inconsistent independent replay')
    return dict(verified=True, configurations=total, origins=451, prefix_records=950, prefix_bytes_requested=29204,
                suffix_requested=False, compared_scalar_metrics=len(differences), verified_blobs=len(verified_blobs),
                maximum_metric_difference=float(maximum), stages=output,
                scope='Independent original-label scoring and frozen gate replay, not retraining, source/vintage certification, or novelty validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=HERE.parents[2])
    options = parser.parse_args()
    print(json.dumps(verify(options.artifact_root, options.repo_root), ensure_ascii=False, indent=2, allow_nan=False))
