"""One fixed 144-fit experiment. Never imports old runners or numerical weights."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import io
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from train_state import MODES, SEEDS, configure, fit, predict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX_SHA = '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_prefix():
    source = ROOT / 'data/parsed/figshare_28682348/guangzhou_reported.csv'
    with source.open('rb', buffering=0) as handle:
        raw = handle.read(29204)
    if len(raw) != 29204 or hashlib.sha256(raw).hexdigest() != PREFIX_SHA:
        raise ValueError('Approved prefix identity changed')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(rows) != 950 or rows[-1]['date'] != '2021-12-31':
        raise ValueError('Approved prefix boundary changed')
    prices = np.array([float(row['author_reported_price']) for row in rows])
    dates = np.array([row['date'] for row in rows])
    if not np.isfinite(prices).all() or not np.all(dates[1:] > dates[:-1]):
        raise ValueError('Invalid prefix values or order')
    return prices, dates


def summarize(prediction, truth, years):
    ratios = []
    ensembles = []
    cells = []
    annual = {}
    for year in ['2020', '2021']:
        mask = years == year
        year_cells = []
        for horizon in [1, 5, 20]:
            y = truth[mask, horizon - 1]
            denom = np.mean(y ** 2)
            error = prediction[:, mask, horizon - 1] - y[None]
            ratio = np.mean(error ** 2, axis=1) / denom
            ratios.append(ratio)
            year_cells.append(float(ratio.mean()))
            ensembles.append(float(np.mean((prediction[:, mask, horizon - 1].mean(0) - y) ** 2) / denom))
            cells.append(dict(year=year, horizon=horizon, count=int(mask.sum()), ratio=float(ratio.mean())))
        annual[year] = float(np.mean(year_cells))
    seeds = np.mean(ratios, axis=0)
    return dict(score=float(seeds.mean()), ensemble=float(np.mean(ensembles)), seeds=seeds.tolist(), annual=annual, cells=cells)


def trajectory(args):
    mode, seed, price = args
    configure()
    local = HERE / 'local_only'
    origins = np.arange(459, 930)
    prediction = np.full((471, 20), np.nan)
    logs = []
    for block in range(12):
        end = 459 + 40 * block
        selected = np.flatnonzero((origins - 459) // 40 == block)
        start = time.perf_counter()
        model, scale, log = fit(price, end, mode, seed)
        forecast = predict(model, scale, price, origins[selected], mode)
        prediction[selected] = forecast
        log.update(block=block, seconds=time.perf_counter() - start)
        key = f'{mode}_s{seed}_b{block:02d}'
        numeric = {'state__' + name: value.detach().numpy() for name, value in model.state_dict().items()}
        numeric.update(prediction=forecast, origins=origins[selected], scale=np.array(scale))
        np.savez_compressed(local / (key + '.npz'), **numeric)
        write_json(local / (key + '.json'), log)
        logs.append(log)
        print(f'{key} epochs={log["epochs"]} selected={log["selected_epoch"]}', flush=True)
    if not np.isfinite(prediction).all():
        raise ValueError('Incomplete trajectory')
    np.savez_compressed(local / f'{mode}_s{seed}.npz', prediction=prediction)
    return mode, seed, prediction, logs


def main(workers):
    local = HERE / 'local_only'
    if (HERE / 'FIT_LOGS.json').exists() or (local.exists() and any(local.glob('*.npz'))):
        raise FileExistsError('Preserve the first run; use a separately frozen directory for any new experiment')
    local.mkdir(exist_ok=True)
    price, dates = load_prefix()
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(HERE.iterdir()) if p.suffix == '.py' or p.name == 'PROTOCOL.md'}
    write_json(HERE / 'PRE_RUN.json', dict(source_sha256=source_hashes, records=950, prefix_bytes=29204,
               prefix_sha256=PREFIX_SHA, boundary_requested=False, workers=workers, expected_fits=144))
    write_json(HERE / 'ENVIRONMENT.json', dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
               platform=platform.platform(), dtype='float64', single_thread=True, mkldnn=False, workers=workers))
    start = time.perf_counter()
    logs = []
    predictions = {}
    tasks = [(mode, seed, price) for mode in MODES for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(trajectory, task) for task in tasks]):
            mode, seed, prediction, records = future.result()
            predictions[(mode, seed)] = prediction
            logs.extend(records)
            write_json(HERE / 'FIT_LOGS.json', logs)
    formal_seconds = time.perf_counter() - start
    origins = np.arange(459, 930)
    eligible = dates[origins].astype('U4') == dates[origins + 20].astype('U4')
    scored_origins = origins[eligible]
    truth = np.array([price[t + 1:t + 21] - price[t] for t in scored_origins])
    years = dates[scored_origins].astype('U4')
    scores = {}
    for mode in MODES:
        pr = np.stack([predictions[(mode, seed)][eligible] for seed in SEEDS])
        np.savez_compressed(local / (mode + '.npz'), prediction=pr)
        scores[mode] = summarize(pr, truth, years)
    candidate = scores['full_prequential_mature']
    gains = {mode: 1 - candidate['score'] / score['score'] for mode, score in scores.items()}
    wins = {mode: int(np.sum(np.array(candidate['seeds']) < np.array(score['seeds']))) for mode, score in scores.items()}
    passed = all(gains[name] >= .01 and wins[name] == 2 for name in ['full_profile', 'full_prequential_complete'])
    passed = passed and all(candidate['annual'][y] <= scores['full_profile']['annual'][y] for y in ['2020', '2021'])
    passed = passed and candidate['score'] <= .98
    result = dict(complete=True, formal_fits=len(logs), formal_wall_seconds=formal_seconds, n_origins=len(scored_origins),
                  scores=scores, candidate_gains=gains, candidate_seed_wins=wins, candidate_gate=bool(passed),
                  novelty_established=False, unseen_validation=False,
                  status='CANDIDATE_DEVELOPMENT_GATE_ONLY' if passed else 'STOP_FIXED_PREQUENTIAL_RULE')
    write_json(HERE / 'RESULTS.json', result)
    causal = []
    for mode in ['full_joint', 'full_profile', 'full_prequential_complete', 'full_prequential_mature']:
        altered = price.copy()
        altered[459:] += 10000
        t0 = time.perf_counter()
        first, a_scale, _ = fit(price, 459, mode, 17)
        changed, b_scale, _ = fit(altered, 459, mode, 17)
        equal = a_scale == b_scale and all(torch.equal(value, changed.state_dict()[key]) for key, value in first.state_dict().items())
        a_pred = predict(first, a_scale, price, [458], mode)
        b_pred = predict(changed, b_scale, altered, [458], mode)
        with np.load(local / f'{mode}_s17_b00.npz', allow_pickle=False) as saved:
            original_equal = all(np.array_equal(value.detach().numpy(), saved['state__' + key]) for key, value in first.state_dict().items())
        if not equal or not original_equal or not np.array_equal(a_pred, b_pred):
            raise ValueError('Representative full refit causality/reproduction failed: ' + mode)
        causal.append(dict(mode=mode, seed=17, end=459, perturb_from=459, parameters_equal=True,
                           first_fit_equal=True, predictions_equal=True, actual_fits=2, seconds=time.perf_counter() - t0))
        write_json(HERE / 'CAUSAL.json', causal)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    main(parser.parse_args().workers)
