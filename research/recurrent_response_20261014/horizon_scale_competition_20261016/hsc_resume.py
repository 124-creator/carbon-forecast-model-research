"""Recover saved fresh-fit blocks without resetting the original deadline."""
import datetime
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from hsc_run import HERE, CUTS, SEEDS, MODES, configurations, source_hashes, dump, save_fit, market_metrics
from hsc_fit import fit, predict


def missing_tasks(root, selection, grid, modes=MODES, cuts=CUTS, seeds=SEEDS):
    root = Path(root)
    missing = []
    for mode in modes:
        cid = selection[mode]['config_id']
        for seed in seeds:
            for cut in cuts:
                stem = root / f'{mode}_{cid}_s{seed}_t{cut}'
                a, b = stem.with_suffix('.json').exists(), stem.with_suffix('.npz').exists()
                if a != b:
                    raise ValueError('Partial saved fit pair: ' + stem.name)
                if not a:
                    missing.append((mode, grid[cid], seed, cut))
    return missing


def one(task):
    mode, config, seed, cut, root, deadline = task
    if time.monotonic() >= deadline:
        raise TimeoutError('Original deadline expired')
    with np.load(Path(root) / 'INPUTS.npz', allow_pickle=False) as z:
        price = z['price']
    model, scale, log = fit(price, cut, mode, config, seed, deadline=deadline)
    ids = np.arange(cut, min(cut + 40, 930))
    prediction, weights = predict(model, scale, price, ids)
    log.update(phase='market', config_id=config['id'], cut=cut, origin_first=int(ids[0]), origin_last=int(ids[-1]))
    stem = f'{mode}_{config["id"]}_s{seed}_t{cut}'
    save_fit(Path(root) / 'market_fits', stem, model, scale, log, prediction, weights, ids)
    print('RECOVERED ' + stem, flush=True)
    return stem


def resume():
    root = HERE / 'local_only'
    before = json.loads((HERE / 'PRE_RUN.json').read_text(encoding='utf-8'))
    if source_hashes() != before['source_sha256']:
        raise ValueError('Frozen source changed')
    if (HERE / 'RUN_COMPLETE.json').exists() or (HERE / 'INCOMPLETE.json').exists():
        raise ValueError('No repeated completion or expired-budget recovery')
    receipt_path = HERE / 'RESUME_RECEIPT.json'
    if receipt_path.exists():
        old_receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if old_receipt['source_sha256'] != before['source_sha256']:
            raise ValueError('Earlier recovery sources differ')
        receipt_path = HERE / 'RESUME_AFTER_TOOL_TIMEOUT.json'
        if receipt_path.exists():
            raise ValueError('Only one tool-timeout recovery is allowed')
    selection = json.loads((HERE / 'SELECTION.json').read_text(encoding='utf-8'))
    grid = {c['id']: c for c in configurations()}
    start = datetime.datetime.fromisoformat(before['started_at'])
    now = datetime.datetime.now(datetime.timezone.utc)
    remaining = 1800 - (now - start).total_seconds()
    if remaining <= 0:
        raise TimeoutError('Original wall deadline expired before recovery')
    missing = missing_tasks(root / 'market_fits', selection, grid)
    # Change only scheduling order: distribute candidate/control blocks first.
    priority = {'horizon_competition': 0, 'shared_competition': 1, 'gate_only': 2, 'scale_mean': 3, 'raw_gru': 4, 'raw_lstm': 5}
    missing.sort(key=lambda t: (t[3], t[2], priority[t[0]]))
    dump(receipt_path, dict(original_exit=127, cause='Abrupt process exit then terminal-tool420-second timeout; no scientific-result verdict',
         original_saved_search=48, original_saved_market=144-len(missing), missing_fits=len(missing),
         remaining_wall_seconds=remaining, source_sha256=before['source_sha256'],
         resumed_at=now.isoformat(), deadline_reset=False, existing_saved_fits_reused=True))
    deadline = time.monotonic() + remaining
    try:
        with ProcessPoolExecutor(max_workers=3) as pool:
            tasks = [(*t, str(root), deadline) for t in missing]
            for f in as_completed([pool.submit(one, t) for t in tasks]):
                f.result()
    except Exception as error:
        dump(HERE / 'INCOMPLETE.json', dict(error=type(error).__name__, reason=str(error),
             market_saved=len(list((root / 'market_fits').glob('*.json'))), complete=False, resumed=True))
        raise
    if source_hashes() != before['source_sha256']:
        raise ValueError('Frozen sources differ after recovery')
    with np.load(root / 'INPUTS.npz', allow_pickle=False) as z:
        price, dates = z['price'], z['dates']
    results, diagnostics = {}, {}
    for mode in MODES:
        cid = selection[mode]['config_id']
        curves, weights = [], []
        for seed in SEEDS:
            parts, weight_parts = [], []
            for cut in CUTS:
                with np.load(root / 'market_fits' / f'{mode}_{cid}_s{seed}_t{cut}.npz', allow_pickle=False) as z:
                    parts.append(z['prediction'])
                    weight_parts.append(z['weights'])
            curves.append(np.concatenate(parts))
            weights.append(np.concatenate(weight_parts))
        results[mode] = market_metrics(price, dates, np.arange(459, 930), np.stack(curves))
        mean = np.mean(weights, axis=(0, 1))
        diagnostics[mode] = dict(mean_weights_by_scale_horizon=mean.tolist(), horizon_weight_range=float(np.max(np.ptp(mean, axis=1))))
    counts = dict(search=len(list((root / 'search_fits').glob('*.json'))), market=len(list((root / 'market_fits').glob('*.json'))))
    if counts != dict(search=48, market=144):
        raise ValueError('Incomplete fit count')
    dump(HERE / 'RESULTS.json', dict(complete=True, models=results, routing=diagnostics, status='DEVELOPMENT_ONLY', unseen_test=False, novelty_established=False))
    dump(HERE / 'RUN_COMPLETE.json', dict(complete=True, formal_fits=192, fits=counts,
         seconds=(datetime.datetime.now(datetime.timezone.utc)-start).total_seconds(), source_sha256=before['source_sha256'], resumed=True))
    print(json.dumps({k:v['score'] for k,v in results.items()}, indent=2), flush=True)


if __name__ == '__main__':
    resume()
