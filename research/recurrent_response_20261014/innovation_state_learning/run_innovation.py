"""Single fixed market run; no resume, tuning, old predictions or checkpoints."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
import json
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from experiment_io import HERE, load_prefix, summarize, write_json
from train_innovation import MODES, configure, fit, predict


def planned_jobs():
    return [(mode, seed) for mode in MODES for seed in ((17,) if mode == 'constant_shift' else (17, 29))]


def trajectory(task):
    mode, seed, price, deadline = task
    configure()
    local = HERE / 'local_only'
    origins = np.arange(459, 930)
    prediction = np.full((len(origins), 20), np.nan)
    logs = []
    for block in range(12):
        end = 459 + 40 * block
        chosen = np.flatnonzero((origins - 459) // 40 == block)
        start = time.perf_counter()
        model, scale, log = fit(price, end, mode, seed, deadline=deadline)
        forecast = predict(model, scale, price, origins[chosen], mode)
        prediction[chosen] = forecast
        log.update(block=block, seconds=time.perf_counter() - start)
        key = f'{mode}_s{seed}_b{block:02d}'
        saved = {'state__' + name: value.detach().numpy() for name, value in model.state_dict().items()}
        saved.update(prediction=forecast, origins=origins[chosen], scale=np.array(scale))
        np.savez_compressed(local / (key + '.npz'), **saved)
        write_json(local / (key + '.json'), log)
        logs.append(log)
        print(f'{key} epochs={log["epochs"]} chosen={log["selected_epoch"]}', flush=True)
    if not np.isfinite(prediction).all():
        raise ValueError('Incomplete trajectory')
    np.savez_compressed(local / f'{mode}_s{seed}.npz', prediction=prediction)
    return mode, seed, prediction, logs


def causal_refits(price, deadline):
    records = []
    changed = price.copy()
    changed[459:] += 10000
    for mode in ('innovation_shift', 'gru_profile', 'gru_rank_window'):
        first, scale, log = fit(price, 459, mode, 17, deadline=deadline)
        other, scale2, log2 = fit(changed, 459, mode, 17, deadline=deadline)
        if scale != scale2 or log != log2:
            raise ValueError('Training suffix changed scale/trace: ' + mode)
        for name, value in first.state_dict().items():
            if not torch.equal(value, other.state_dict()[name]):
                raise ValueError('Training suffix changed parameters: ' + mode)
        a = predict(first, scale, price, np.array([458]), mode)
        b = predict(other, scale2, changed, np.array([458]), mode)
        if not np.array_equal(a, b):
            raise ValueError('Training suffix changed before-cutoff forecast')
        with np.load(HERE / 'local_only' / f'{mode}_s17_b00.npz', allow_pickle=False) as original:
            if not all(np.array_equal(value.detach().numpy(), original['state__' + name]) for name, value in first.state_dict().items()):
                raise ValueError('First formal fit did not reproduce: ' + mode)
        records.append(dict(mode=mode, exclusive_end=459, perturb_from=459, full_refits=2,
                            scale_trace_parameters_predictions_equal=True, first_formal_fit_reproduced=True))
        write_json(HERE / 'CAUSAL.json', records)
    return records


def main(workers):
    configure()
    local = HERE / 'local_only'
    if local.exists() or (HERE / 'PRE_RUN.json').exists():
        raise FileExistsError('Preserve the first run; no resume or silent overwrite')
    local.mkdir()
    price, dates, receipt = load_prefix()
    write_json(HERE / 'INPUT_AUDIT.json', receipt)
    write_json(HERE / 'ENVIRONMENT.json', dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
              platform=platform.platform(), dtype='float64', threads=1, mkldnn=False, workers=workers))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix == '.py' or p.name == 'PROTOCOL.md'}
    jobs = planned_jobs()
    write_json(HERE / 'PRE_RUN.json', dict(source_sha256=hashes, jobs=jobs, expected_market_fits=len(jobs) * 12,
               market_and_refit_budget_seconds=1200, representative_boundary_refits=6, unseen_validation=False))
    deadline = time.time() + 1200
    start = time.perf_counter()
    logs, predictions = [], {}
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(trajectory, (mode, seed, price, deadline)) for mode, seed in jobs]
        for future in as_completed(futures):
            mode, seed, prediction, records = future.result()
            predictions[(mode, seed)] = prediction
            logs.extend(records)
            write_json(HERE / 'FIT_LOGS.json', logs)
        pool.shutdown(wait=True)
        if len(logs) != len(jobs) * 12:
            raise ValueError('Frozen fit total mismatch')
        origins = np.arange(459, 930)
        eligible = dates[origins].astype('U4') == dates[origins + 20].astype('U4')
        scored = origins[eligible]
        if len(scored) != 451:
            raise ValueError('Common-origin count changed')
        truth = np.array([price[t + 1:t + 21] - price[t] for t in scored])
        years = dates[scored].astype('U4')
        scores = {}
        for mode in MODES:
            seeds = (17,) if mode == 'constant_shift' else (17, 29)
            pr = np.stack([predictions[(mode, seed)][eligible] for seed in seeds])
            np.savez_compressed(local / (mode + '.npz'), prediction=pr)
            scores[mode] = summarize(pr, truth, years)
        ordinary = ('constant_shift', 'gru_profile', 'gru_rank_window', 'lstm_profile')
        best = min(ordinary, key=lambda mode: scores[mode]['score'])
        candidate = scores['innovation_shift']
        improvement = 1 - candidate['score'] / scores[best]['score']
        wins = int(np.sum(np.array(candidate['seeds']) < np.array(scores[best]['seeds'])))
        cell_wins = sum(a['ratio'] < b['ratio'] for a, b in zip(candidate['cells'], scores[best]['cells']))
        ablations = {mode: candidate['score'] < scores[mode]['score'] for mode in ('raw_shift', 'innovation_diagonal', 'constant_shift')}
        gate = improvement >= .01 and wins == 2 and cell_wins >= 4 and all(ablations.values())
        causal = causal_refits(price, deadline)
        result = dict(complete=True, market_fits=len(logs), boundary_refits=sum(r['full_refits'] for r in causal),
                      wall_seconds=time.perf_counter() - start, n_origins=len(scored), scores=scores,
                      best_ordinary=best, improvement_over_best=improvement, paired_seed_wins=wins,
                      six_cell_wins=cell_wins, ablation_gates=ablations, candidate_gate=bool(gate),
                      novelty_established=False, paper_fulltext_gate=False, unseen_validation=False,
                      status='DEVELOPMENT_CANDIDATE_ONLY' if gate else 'STOP_FIXED_FORECAST_STATE_RULE')
        write_json(HERE / 'RESULTS.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        for future in locals().get('futures', []):
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        failure = traceback.format_exc()
        write_json(HERE / 'INCOMPLETE.json', dict(complete=False, wall_seconds=time.perf_counter() - start,
                   completed_trajectory_fits=len(logs), candidate_gate=False, error=failure))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    main(parser.parse_args().workers)
