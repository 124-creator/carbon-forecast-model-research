"""One frozen CPU run. Keep every interrupted block; never resume or overwrite."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import torch
from event_model import MODES, EventNet
from experiment_io import HERE, summarize, write_json
from reference_support import configure
from train_events import fit, predict

LOCAL = HERE / 'local_only'
RUNS = LOCAL / 'runs'


def plan(dates):
    calendar = np.asarray(dates).astype('datetime64[D]').astype(str)
    years = np.array([s[:4] for s in calendar])
    start = int(np.flatnonzero(years == '2020')[0])
    origins = np.array([t for t in range(start, len(dates) - 20)
                        if years[t] in ('2020', '2021') and years[t] == years[t + 20]], dtype=np.int64)
    blocks = [dict(end=end, origins=origins[(origins >= end) & (origins < end + 40)])
              for end in range(start, len(dates) - 20, 40)]
    blocks = [block for block in blocks if len(block['origins'])]
    jobs = [(mode, seed) for mode in ('event', 'hold', 'price', 'clock', 'raw_hold', 'raw_event') for seed in (17, 29)]
    return origins, blocks, jobs


def load_data():
    expected = json.loads((HERE / 'DATA_AUDIT.json').read_text(encoding='utf-8'))['local_bundle_sha256']
    if hashlib.sha256((LOCAL / 'INPUTS.npz').read_bytes()).hexdigest() != expected:
        raise ValueError('Prepared input identity changed')
    with np.load(LOCAL / 'INPUTS.npz', allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def trajectory(task):
    mode, seed, deadline = task
    configure()
    data = load_data()
    origins, blocks, _ = plan(data['dates'])
    predictions, logs = [], []
    for block, item in enumerate(blocks):
        started = time.perf_counter()
        model, stats, log = fit(data, item['end'], mode, seed, deadline=deadline)
        forecast = predict(model, stats, data, item['origins'])
        key = f'{mode}_s{seed}_b{block:02d}'
        state = {'state__' + name: value.detach().numpy() for name, value in model.state_dict().items()}
        np.savez_compressed(RUNS / (key + '.npz'), **state, prediction=forecast, origins=item['origins'])
        log.update(block=block, seconds=time.perf_counter() - started)
        write_json(RUNS / (key + '.json'), log)
        logs.append(log)
        predictions.append(forecast)
        print(f'{key} epochs={log["epochs"]} seconds={log["seconds"]:.2f}', flush=True)
    prediction = np.concatenate(predictions)
    if prediction.shape != (len(origins), 20) or not np.isfinite(prediction).all():
        raise ValueError('Incomplete or nonfinite trajectory')
    np.savez_compressed(RUNS / f'{mode}_s{seed}.npz', prediction=prediction, origins=origins)
    return mode, seed, prediction, logs


def causal_refits(data, deadline):
    records = []
    for mode in ('event', 'hold'):
        key = f'{mode}_s17_b00'
        original_log = json.loads((RUNS / (key + '.json')).read_text(encoding='utf-8'))
        end = original_log['end']
        with np.load(RUNS / (key + '.npz'), allow_pickle=False) as original:
            state = {name[7:]: original[name].copy() for name in original.files if name.startswith('state__')}
        baseline = EventNet(mode)
        baseline.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
        for kind in ('price', 'source'):
            changed = {k: v.copy() for k, v in data.items()}
            if kind == 'price':
                changed['price'][end:] += 10000
            else:
                changed['source_values'][data['source_dates'] > data['dates'][end - 1]] += 10000
            other, stats, log = fit(changed, end, mode, 17, deadline=deadline)
            if stats != original_log['stats'] or log['history'] != original_log['history']:
                raise ValueError('Future suffix changed training trace or normalization')
            if not all(np.array_equal(state[name], value.detach().numpy()) for name, value in other.state_dict().items()):
                raise ValueError('Future suffix changed fitted parameters')
            if not np.array_equal(predict(baseline, stats, data, [end - 1]), predict(other, stats, changed, [end - 1])):
                raise ValueError('Future suffix changed before-cutoff prediction')
            records.append(dict(mode=mode, kind=kind, seed=17, exclusive_end=end,
                                epochs=log['epochs'], exact_trace_state_scale_prediction_match=True))
            write_json(HERE / 'CAUSAL.json', records)
    return records


def main(workers):
    configure()
    if RUNS.exists() or (HERE / 'PRE_RUN.json').exists():
        raise FileExistsError('Preserve first run. Use a fresh sibling directory for a new run.')
    data = load_data()
    origins, blocks, jobs = plan(data['dates'])
    RUNS.mkdir()
    started = time.time()
    frozen = dict(source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in HERE.iterdir() if p.is_file() and (p.suffix == '.py' or p.name == 'PROTOCOL.md')},
                  jobs=jobs, blocks=[dict(end=b['end'], count=len(b['origins'])) for b in blocks],
                  expected_market_fits=len(jobs) * len(blocks), representative_full_refits=4,
                  market_budget_seconds=1080, total_training_budget_seconds=1200, unseen_validation=False)
    write_json(HERE / 'PRE_RUN.json', frozen)
    write_json(HERE / 'ENVIRONMENT.json', dict(python=platform.python_version(), torch=torch.__version__,
                                             numpy=np.__version__, cuda=torch.cuda.is_available(), workers=workers))
    completed = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(trajectory, (mode, seed, started + 1080)) for mode, seed in jobs]
            for future in as_completed(futures):
                mode, seed, prediction, logs = future.result()
                completed[(mode, seed)] = prediction
                write_json(HERE / 'PROGRESS.json', dict(complete_trajectories=len(completed),
                           saved_completed_fits=len(list(RUNS.glob('*_b*.json'))), expected_trajectories=len(jobs)))
        truth = np.array([data['price'][t + 1:t + 21] - data['price'][t] for t in origins])
        years = np.array([str(np.datetime64(int(data['dates'][t]), 'D'))[:4] for t in origins])
        metrics = {mode: summarize(np.stack([completed[(mode, seed)] for seed in (17, 29)]), truth, years) for mode in MODES}
        write_json(HERE / 'ALL_MODEL_METRICS.json', metrics)
        np.savez_compressed(RUNS / 'EVALUATION.npz', origins=origins, truth=truth, years=years)
        causal = causal_refits(data, started + 1200)
        write_json(HERE / 'COMPLETE.json', dict(complete=True, market_fits=len(jobs) * len(blocks),
                   boundary_refits=len(causal), seconds=time.time() - started, unseen_validation=False,
                   paper_acceptance_passed=False))
        print('COMPLETE ' + json.dumps({m: metrics[m]['score'] for m in MODES}), flush=True)
    except Exception as exc:
        write_json(HERE / 'INCOMPLETE.json', dict(complete=False, acceptance_passed=False,
                   complete_trajectories=len(completed), saved_completed_fits=len(list(RUNS.glob('*_b*.json'))),
                   expected_market_fits=len(jobs) * len(blocks), seconds=time.time() - started,
                   error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args().workers)
