"""Bounded fresh development fits; never rank an incomplete trajectory."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import datetime
import hashlib
import itertools
import json
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from hsc_core import MODES, configure
from hsc_fit import fit, predict
from fair_data import load_prefix

HERE = Path(__file__).resolve().parent
FAIR = HERE.parent / 'fair_recurrent_tuning_20261016'
CUTS = tuple(459 + 40 * b for b in range(12))
SEEDS = (17, 29)


def configurations():
    return [dict(id=f'c{i:02d}', hidden=h, lr=lr, weight_decay=.001)
            for i, (h, lr) in enumerate(itertools.product((8, 16), (.001, .003)))]


def select(records):
    expected = {(m, c['id'], t) for m in MODES for c in configurations() for t in (359, 419)}
    found = {}
    for r in records:
        key = (r['family'], r['config_id'], r['cut'])
        if key not in expected or key in found or not np.isfinite(r['score']) or r['score'] < 0:
            raise ValueError('Unexpected, duplicate or invalid search fit')
        found[key] = r['score']
    if set(found) != expected:
        raise ValueError('Incomplete search cannot select a model')
    result = {}
    for mode in MODES:
        scores = {c['id']: np.mean([found[(mode, c['id'], t)] for t in (359, 419)])
                  for c in configurations()}
        cid = min(scores, key=lambda x: (scores[x], x))
        result[mode] = dict(config_id=cid, score=float(scores[cid]), all_scores=scores)
    return result


def span_metrics(prediction, truth):
    if prediction.ndim != 3 or truth.shape != prediction.shape[1:] or truth.shape[1] != 20:
        raise ValueError('Bad score dimensions')
    if not truth.size or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError('Incomplete or nonfinite forecasts')
    seed_cells, ensembles, cells = [], [], []
    for h in (1, 5, 20):
        actual, forecast = truth[:, h - 1], prediction[:, :, h - 1]
        denominator = float(np.mean(actual ** 2))
        if denominator <= 0:
            raise ValueError('Undefined random-walk denominator')
        errors = forecast - actual
        losses = np.mean(errors ** 2, axis=1) / denominator
        seed_cells.append(losses)
        ensembles.append(float(np.mean((forecast.mean(0) - actual) ** 2) / denominator))
        cells.append(dict(horizon=h, count=len(actual), ratio=float(losses.mean()),
                          rmse=float(np.sqrt(np.mean(errors ** 2))), mae=float(np.mean(np.abs(errors)))))
    seeds = np.mean(seed_cells, axis=0)
    return dict(score=float(seeds.mean()), seeds=seeds.tolist(),
                ensemble=float(np.mean(ensembles)), cells=cells)


def market_metrics(price, dates, origins, predictions):
    ids = np.asarray(origins)
    eligible = np.array([dates[t][:4] == dates[t + 20][:4] for t in ids])
    parts, cells, annual, counts, mape = [], [], {}, {}, {}
    for year in ('2020', '2021'):
        mask = eligible & np.array([dates[t][:4] == year for t in ids])
        selected = ids[mask]
        if not len(selected):
            raise ValueError('Missing primary year')
        truth = np.stack([price[t + 1:t + 21] - price[t] for t in selected])
        result = span_metrics(predictions[:, mask], truth)
        parts.append(result)
        annual[year] = result['score']
        counts[year] = len(selected)
        for c in result['cells']:
            h = c['horizon']
            error = predictions[:, mask, h - 1] - truth[:, h - 1]
            pct = float(np.mean(np.abs(error) / price[selected + h]) * 100)
            mape[f'{year}_h{h}'] = pct
            cells.append(dict(c, year=year, mape_percent=pct))
    seeds = np.mean([p['seeds'] for p in parts], axis=0)
    return dict(score=float(seeds.mean()), seeds=seeds.tolist(),
                ensemble=float(np.mean([p['ensemble'] for p in parts])),
                annual=annual, counts=counts, cells=cells)


def dump(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_hashes():
    paths = [HERE / n for n in ('hsc_core.py', 'hsc_fit.py', 'hsc_run.py', 'PROTOCOL.md')]
    paths += [FAIR / n for n in ('fair_core.py', 'fair_data.py')]
    return {str(p.relative_to(HERE.parent)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def save_fit(root, stem, model, scale, receipt, prediction, weights, origins):
    arrays = {f'param__{k}': v.detach().numpy() for k, v in model.state_dict().items()}
    arrays.update(prediction=prediction, weights=weights, origins=origins, scale=np.array(scale))
    path = Path(root) / (stem + '.npz')
    if path.exists():
        raise FileExistsError('Refuse overwriting a saved fit')
    np.savez_compressed(path, **arrays)
    dump(Path(root) / (stem + '.json'), receipt)


def job(task):
    phase, mode, config, seed, cut, root, deadline = task
    with np.load(Path(root) / ('TUNING_INPUT.npz' if phase == 'search' else 'INPUTS.npz'), allow_pickle=False) as z:
        price = z['price']
    cuts = (cut,) if phase == 'search' else CUTS
    completed = []
    for end in cuts:
        if time.monotonic() >= deadline:
            raise TimeoutError('Frozen run wall budget exhausted')
        origins = np.arange(end, end + 20) if phase == 'search' else np.arange(end, min(end + 40, 930))
        model, scale, log = fit(price, end, mode, config, seed, deadline=deadline)
        forecast, weights = predict(model, scale, price, origins)
        log.update(phase=phase, config_id=config['id'], cut=end,
                   origin_first=int(origins[0]), origin_last=int(origins[-1]))
        if phase == 'search':
            truth = np.stack([price[t + 1:t + 21] - price[t] for t in origins])
            log['score'] = span_metrics(forecast[None], truth)['score']
        stem = f'{mode}_{config["id"]}_s{seed}_t{end}'
        save_fit(Path(root) / (phase + '_fits'), stem, model, scale, log, forecast, weights, origins)
        completed.append(dict(family=mode, config_id=config['id'], seed=seed, cut=end, score=log.get('score')))
        print(f'{phase.upper()} {stem} epoch={log["selected_epoch"]}', flush=True)
    return completed


def timing():
    if (HERE / 'TIMING.json').exists():
        raise FileExistsError('Preserve timing probes')
    price, _, _ = load_prefix()
    records = []
    for mode in MODES:
        started = time.perf_counter()
        _, _, log = fit(price[:459], 459, mode, configurations()[-1], 17, max_epochs=2)
        records.append(dict(family=mode, epochs=log['epochs'], seconds=time.perf_counter() - started))
    dump(HERE / 'TIMING.json', dict(records=records, fits=len(records), scope='Timing, not selection or market score'))
    print(json.dumps(records, indent=2))


def run(workers):
    configure()
    root = HERE / 'local_only'
    if root.exists():
        raise FileExistsError('Fresh experiment only; preserve existing run')
    price, dates, source = load_prefix()
    if dates[458] != '2019-12-31':
        raise ValueError('Search/development date boundary differs')
    root.mkdir()
    for n in ('search_fits', 'market_fits'):
        (root / n).mkdir()
    np.savez_compressed(root / 'TUNING_INPUT.npz', price=price[:459])
    np.savez_compressed(root / 'INPUTS.npz', price=price, dates=dates)
    hashes = source_hashes()
    dump(HERE / 'PRE_RUN.json', dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         modes=list(MODES), configurations=configurations(), search_cuts=[359, 419], market_cuts=list(CUTS),
         seeds=list(SEEDS), formal_expected=192, wall_budget=1800, source_sha256=hashes, data=source,
         workers=workers, unseen_test=False, novelty_established=False))
    dump(HERE / 'ENVIRONMENT.json', dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
         threads=torch.get_num_threads(), dtype=str(torch.get_default_dtype()), cuda=torch.cuda.is_available()))
    start = time.monotonic()
    deadline = start + 1800
    phase = 'search'
    try:
        search = []
        tasks = [('search', mode, c, 17, t, str(root), deadline)
                 for mode in MODES for c in configurations() for t in (359, 419)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(job, task) for task in tasks]):
                search.extend(f.result())
        choice = select(search)
        dump(HERE / 'SEARCH_RESULTS.json', search)
        dump(HERE / 'SELECTION.json', choice)
        grid = {c['id']: c for c in configurations()}
        phase = 'market'
        tasks = [('market', mode, grid[choice[mode]['config_id']], seed, None, str(root), deadline)
                 for mode in MODES for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(job, task) for task in tasks]):
                f.result()
        if source_hashes() != hashes:
            raise ValueError('Frozen source changed during run')
        origins = np.arange(459, 930)
        results, diagnostics = {}, {}
        for mode in MODES:
            curves, weight_rows = [], []
            for seed in SEEDS:
                forecasts, weights = [], []
                cid = choice[mode]['config_id']
                for cut in CUTS:
                    with np.load(root / 'market_fits' / f'{mode}_{cid}_s{seed}_t{cut}.npz', allow_pickle=False) as z:
                        forecasts.append(z['prediction'])
                        weights.append(z['weights'])
                curves.append(np.concatenate(forecasts))
                weight_rows.append(np.concatenate(weights))
            predictions = np.stack(curves)
            results[mode] = market_metrics(price, dates, origins, predictions)
            mean_weights = np.mean(weight_rows, axis=(0, 1))
            diagnostics[mode] = dict(mean_weights_by_scale_horizon=mean_weights.tolist(),
                                    horizon_weight_range=float(np.max(np.ptp(mean_weights, axis=1))))
        counts = dict(search=len(list((root / 'search_fits').glob('*.json'))),
                      market=len(list((root / 'market_fits').glob('*.json'))))
        if counts != dict(search=48, market=144):
            raise ValueError('Formal fit grid is incomplete')
        dump(HERE / 'RESULTS.json', dict(complete=True, models=results, routing=diagnostics,
             status='DEVELOPMENT_ONLY', unseen_test=False, novelty_established=False))
        dump(HERE / 'RUN_COMPLETE.json', dict(complete=True, seconds=time.monotonic() - start,
             fits=counts, formal_fits=sum(counts.values()), source_sha256=hashes))
        print(json.dumps({m: r['score'] for m, r in results.items()}, indent=2))
    except Exception as error:
        dump(HERE / 'INCOMPLETE.json', dict(phase=phase, error=type(error).__name__, reason=str(error),
             seconds=time.monotonic() - start, search_saved=len(list((root / 'search_fits').glob('*.json'))),
             market_saved=len(list((root / 'market_fits').glob('*.json'))), complete=False))
        (root / 'FAILURE.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--timing', action='store_true')
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    if args.workers < 1:
        raise ValueError('Positive workers required')
    if args.timing:
        timing()
    else:
        run(args.workers)
