"""Frozen ordinary recurrent-model tuning and matched rolling comparison."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import datetime
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from protocol import plan, configurations, choose
from fair_core import configure
from fair_data import load_prefix
from fit_recurrent import fit, predict
from scoring import span_score, score

HERE = Path(__file__).resolve().parent

def trajectory_jobs(selection):
    schedule = plan()
    grid = {row['id']: row for row in configurations()}
    jobs, aliases = [], {}
    if set(selection) != set(schedule['families']):
        raise ValueError('Missing family selection')
    for family in schedule['families']:
        selected = selection[family]['config_id']
        aliases[family] = dict(default=schedule['default_id'], selected=selected)
        for cid in dict.fromkeys(aliases[family].values()):
            if cid not in grid:
                raise ValueError('Unknown selected configuration')
            for seed in schedule['market_seeds']:
                jobs.append(dict(family=family, config=grid[cid], seed=seed,
                                 key=f'{family}_{cid}_s{seed}'))
    return jobs, aliases


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def save_fit(root, stem, model, scale, receipt, prediction, origins):
    arrays = {f'param__{name}': value.detach().numpy() for name, value in model.state_dict().items()}
    arrays.update(prediction=prediction, origins=origins, scale=np.array(scale))
    np.savez_compressed(Path(root) / (stem + '.npz'), **arrays)
    write_json(Path(root) / (stem + '.json'), receipt)


def search_job(task):
    root, family, config, cut, deadline = task
    if time.monotonic() >= deadline:
        raise TimeoutError('Search budget exhausted before fit')
    with np.load(Path(root) / 'TUNING_INPUT.npz', allow_pickle=False) as data:
        price = data['price']
    schedule = plan()
    started = time.perf_counter()
    model, scale, receipt = fit(price, cut, family, config, schedule['search_seed'],
                                 max_epochs=schedule['max_epochs'], deadline=deadline)
    origins = np.arange(cut, cut + schedule['search_span'])
    forecast = predict(model, scale, price, origins)
    truth = np.stack([price[t + 1:t + 21] - price[t] for t in origins])
    metrics = span_score(forecast[None], truth)
    receipt.update(phase='search', config_id=config['id'], cut=cut,
                   selection_score=metrics['score'], outer_max_target=int(origins[-1] + 20),
                   job_seconds=time.perf_counter() - started)
    save_fit(Path(root) / 'search_fits', f'{family}_{config["id"]}_t{cut}',
             model, scale, receipt, forecast, origins)
    return dict(family=family, config_id=config['id'], cut=cut, selection_score=metrics['score'])


def market_job(task):
    job, root, deadline = task
    if time.monotonic() >= deadline:
        raise TimeoutError('Market budget exhausted before trajectory')
    with np.load(Path(root) / 'INPUTS.npz', allow_pickle=False) as data:
        price = data['price']
    schedule = plan()
    origins = np.asarray(schedule['market_origins'])
    prediction = np.full((len(origins), 20), np.nan)
    for block, cut in enumerate(schedule['market_cuts']):
        selection = np.flatnonzero((origins - 459) // 40 == block)
        model, scale, receipt = fit(price, cut, job['family'], job['config'], job['seed'],
                                    max_epochs=schedule['max_epochs'], deadline=deadline)
        forecast = predict(model, scale, price, origins[selection])
        prediction[selection] = forecast
        receipt.update(phase='market', key=job['key'], block=block,
                       origin_min=int(origins[selection].min()), origin_max=int(origins[selection].max()))
        save_fit(Path(root) / 'market_fits', f'{job["key"]}_b{block:02d}',
                 model, scale, receipt, forecast, origins[selection])
        print(f'MARKET {job["key"]} block={block} selected_epoch={receipt["selected_epoch"]}', flush=True)
    if not np.isfinite(prediction).all():
        raise ValueError('Incomplete forecast trajectory')
    np.savez_compressed(Path(root) / 'curves' / (job['key'] + '.npz'), prediction=prediction)
    return dict(key=job['key'], fit_count=len(schedule['market_cuts']))


def timing_report():
    target = HERE / 'TIMING.json'
    if target.exists():
        raise FileExistsError('Preserve the first timing probes')
    price, dates, _ = load_prefix()
    config = configurations()[-1]
    records = []
    for family in plan()['families']:
        started = time.perf_counter()
        _, _, receipt = fit(price[:459], 459, family, config, 17, max_epochs=2)
        records.append(dict(family=family, epochs=receipt['epochs'],
                            full_fit_seconds=time.perf_counter() - started))
    write_json(target, dict(probe_fits=len(records), price_cut=459,
                            max_input_date=str(dates[458]), records=records,
                            scope='Timing only; frozen grid/epochs unchanged; no market score'))
    print(json.dumps(records), flush=True)


def run(workers):
    configure()
    root = HERE / 'local_only'
    if root.exists():
        raise FileExistsError('Fresh run only; preserve existing artifacts')
    root.mkdir()
    for name in ('search_fits', 'market_fits', 'curves'):
        (root / name).mkdir()
    price, dates, data_receipt = load_prefix()
    if dates[458] != '2019-12-31' or dates[459][:4] != '2020':
        raise ValueError('Search/development boundary differs')
    np.savez_compressed(root / 'INPUTS.npz', price=price, dates=dates)
    np.savez_compressed(root / 'TUNING_INPUT.npz', price=price[:459])
    sources = ['protocol.py', 'fair_core.py', 'fair_data.py', 'fit_recurrent.py',
               'run_fair.py', 'scoring.py', 'PROTOCOL.md']
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in sources}
    schedule = plan()
    write_json(HERE / 'PRE_RUN.json', dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               source_sha256=hashes, plan=schedule, configurations=configurations(), data=data_receipt,
               workers=workers, search_market_boundary=459, status='ORDINARY_DIAGNOSTIC_NOT_INNOVATION'))
    write_json(HERE / 'ENVIRONMENT.json', dict(python=sys.version, numpy=np.__version__,
               torch=torch.__version__, workers=workers, math_threads=torch.get_num_threads(),
               mkldnn=torch.backends.mkldnn.enabled, dtype=str(torch.get_default_dtype())))
    started = time.monotonic()
    deadline = started + schedule['fit_budget_seconds']
    records, completed = [], []
    jobs, aliases = [], {}
    phase = 'search'
    try:
        tasks = [(str(root), family, config, cut, deadline) for family in schedule['families']
                 for config in configurations() for cut in schedule['search_cuts']]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(search_job, task) for task in tasks]):
                records.append(future.result())
                write_json(root / 'SEARCH_PROGRESS.json', records)
                print(f'SEARCH completed={len(records)}/{len(tasks)}', flush=True)
        selection = choose(records)
        write_json(HERE / 'TUNING_RESULTS.json', records)
        write_json(HERE / 'SELECTION.json', selection)
        if any(hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise ValueError('Frozen execution sources changed during search')
        jobs, aliases = trajectory_jobs(selection)
        write_json(HERE / 'MARKET_PLAN.json', dict(jobs=jobs, aliases=aliases,
                   unique_trajectories=len(jobs), expected_unique_market_fits=len(jobs) * 12))
        phase = 'market'
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(market_job, (job, str(root), deadline)) for job in jobs]):
                completed.append(future.result())
                write_json(root / 'TRAJECTORIES.json', completed)
    except Exception as error:
        write_json(HERE / 'INCOMPLETE.json', dict(phase=phase, exception=type(error).__name__,
                   reason=str(error), seconds=time.monotonic() - started,
                   saved_search_fits=len(list((root / 'search_fits').glob('*.json'))),
                   saved_market_fits=len(list((root / 'market_fits').glob('*.json'))),
                   complete_trajectories=len(list((root / 'curves').glob('*.npz'))),
                   expected_market_fits=len(jobs) * 12 if jobs else None,
                   qualification_pass=False))
        (root / 'FAILURE.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    origins = np.asarray(schedule['market_origins'])
    eligible = np.array([dates[t][:4] == dates[t + 20][:4] for t in origins])
    truth = np.stack([price[t + 1:t + 21] - price[t] for t in origins])[eligible]
    years = np.array([dates[t][:4] for t in origins])[eligible]
    models = {}
    for family in schedule['families']:
        for variant, cid in aliases[family].items():
            arrays = []
            for seed in schedule['market_seeds']:
                with np.load(root / 'curves' / f'{family}_{cid}_s{seed}.npz', allow_pickle=False) as data:
                    arrays.append(data['prediction'])
            models[f'{family}__{variant}'] = score(np.stack(arrays)[:, eligible], truth, years)
    write_json(HERE / 'RESULTS.json', dict(status='COMPLETE_DIAGNOSTIC_NOT_INNOVATION', complete=True,
               models=models, aliases=aliases, unseen_test=False, novelty_established=False))
    write_json(HERE / 'RUN_COMPLETE.json', dict(complete=True, seconds=time.monotonic() - started,
               search_fits=len(list((root / 'search_fits').glob('*.json'))),
               market_fits=len(list((root / 'market_fits').glob('*.json'))),
               complete_trajectories=len(completed), unique_jobs=len(jobs), source_sha256=hashes))
    print(json.dumps({name: value['score'] for name, value in models.items()}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--timing', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('Worker count must be positive')
    if args.timing:
        timing_report()
    else:
        run(args.workers)
