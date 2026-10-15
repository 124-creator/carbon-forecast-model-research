import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import copy
import importlib.util
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
from model import Model

R = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ordinary_recurrent_reference', R.parent / 'run.py')
if spec is None or spec.loader is None:
    raise RuntimeError('Cannot locate reviewed ordinary recurrent implementation')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
KINDS = ['GRUpatch', 'RKNglobal', 'RKNconditional']


def fit(p, end, kind, seed, max_epochs=200):
    ids, x, y, tr, va, sd = base.arrays(p, end)
    torch.manual_seed(seed)
    model = Model(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    xx = torch.tensor(x / sd, dtype=torch.float32)
    yy = torch.tensor(y / sd, dtype=torch.float32)
    with torch.no_grad():
        best = float((model(xx[va]) - yy[va]).square().mean())
    state = copy.deepcopy(model.state_dict())
    selected, wait, history = 0, 0, []
    for step in range(max_epochs):
        optimizer.zero_grad()
        loss = (model(xx[tr]) - yy[tr]).square().mean()
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1))
        optimizer.step()
        with torch.no_grad():
            validation = float((model(xx[va]) - yy[va]).square().mean())
        history.append([float(loss.detach()), validation, norm])
        if not np.isfinite(history[-1]).all():
            raise ValueError('Non-finite training')
        if validation < best - 1e-6:
            best, state, selected, wait = validation, copy.deepcopy(model.state_dict()), step + 1, 0
        else:
            wait += 1
        if wait >= 20:
            break
    model.load_state_dict(state)
    parameters = sum(value.numel() for value in model.parameters())
    return model, sd, dict(epochs=len(history), selected=selected, history=history, n=len(ids),
        max_training_target=int(ids[tr[-1]] + 20), validation_first=int(ids[va[0]]),
        max_validation_target=int(ids[va[-1]] + 20), parameters=parameters,
        active_parameters=parameters - (128 if kind == 'RKNglobal' else 0))


def predict(model, sd, p, origins):
    x = np.array([np.diff(p[t-60:t+1]) for t in origins])
    with torch.no_grad():
        return model(torch.tensor(x / sd, dtype=torch.float32)).numpy().astype(float) * sd


def summarize(saved):
    scores = {}
    truth, years = saved['truth'], saved['years']
    for kind in KINDS:
        if not all(kind + '_' + str(seed) in saved and np.isfinite(saved[kind + '_' + str(seed)]).all() for seed in [17, 29]):
            continue
        cells, year_scores, seed_scores = [], {}, {}
        for seed in [17, 29]:
            ratios = []
            for year in ['2020', '2021']:
                mask = years == year
                for h in [1, 5, 20]:
                    denominator = float(np.mean(truth[mask, h-1] ** 2))
                    if denominator <= 0:
                        raise ValueError('Invalid RW denominator')
                    ratios.append(float(np.mean((saved[kind + '_' + str(seed)][mask, h-1] - truth[mask, h-1]) ** 2) / denominator))
            seed_scores[str(seed)] = float(np.mean(ratios))
        for year in ['2020', '2021']:
            mask = years == year
            for h in [1, 5, 20]:
                denominator = np.mean(truth[mask, h-1] ** 2)
                mse = np.mean([np.mean((saved[kind + '_' + str(seed)][mask, h-1] - truth[mask, h-1]) ** 2) for seed in [17, 29]])
                cells.append(dict(year=year, h=h, ratio=float(mse / denominator)))
            year_scores[year] = float(np.mean([c['ratio'] for c in cells if c['year'] == year]))
        scores[kind] = dict(score=float(np.mean([c['ratio'] for c in cells])), cells=cells,
                            year_scores=year_scores, seed_scores=seed_scores)
    gate = len(scores) == 3
    if gate:
        target = scores['RKNconditional']
        gate = all(target['score'] < scores[k]['score'] and all(target['year_scores'][y] <= scores[k]['year_scores'][y] for y in ['2020', '2021']) for k in ['RKNglobal', 'GRUpatch'])
    return dict(scores=scores, development_gate=bool(gate))


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if (R / 'FIT_LOGS.json').exists() or (R / 'local_only/PREDICTIONS.npz').exists():
        raise RuntimeError('Existing run preserved; no implicit rerun or partial resume')
    source_files = [R / 'PROTOCOL.md', R / 'model.py', Path(__file__), base.R / 'run.py']
    receipt = dict(status='before_market_fit', source_sha256={str(p.relative_to(R.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    (R / 'PRE_RUN.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    rows, audit = base.read_prefix(base.R / '../../data/parsed/figshare_28682348/guangzhou_reported.csv', 950, '2021-12-31')
    p = np.array([float(row['author_reported_price']) for row in rows])
    dates = np.array([row['date'] for row in rows])
    origins = np.array([t for t in range(459, 930) if dates[t][:4] == dates[t+20][:4]])
    if len(origins) != 451:
        raise ValueError('Unexpected complete-year origin support')
    saved = dict(origins=origins, truth=np.array([p[t+1:t+21] - p[t] for t in origins]), years=np.array([d[:4] for d in dates[origins]]))
    (R / 'INPUT_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    local = R / 'local_only'
    local.mkdir(exist_ok=True)
    logs, causal, elapsed_total = [], [], 0.
    for block in range(12):
        end = 459 + 40 * block
        indices = np.where((origins - 459) // 40 == block)[0]
        for seed in [17, 29]:
            for kind in KINDS:
                if elapsed_total > 800:
                    break
                key = kind + '_' + str(seed)
                saved.setdefault(key, np.full(saved['truth'].shape, np.nan))
                start = time.perf_counter()
                model, sd, row = fit(p, end, kind, seed)
                saved[key][indices] = predict(model, sd, p, origins[indices])
                seconds = time.perf_counter() - start
                elapsed_total += seconds
                np.savez_compressed(local / (key + '_block' + str(block) + '_parameters.npz'), **{k: v.numpy() for k, v in model.state_dict().items()})
                logs.append(dict(kind=kind, seed=seed, block=block, fit_end=end, seconds=seconds, **row))
                np.savez_compressed(local / 'PREDICTIONS.npz', **saved)
                (R / 'FIT_LOGS.json').write_text(json.dumps(logs), encoding='utf-8')
                print(kind, seed, block, row['epochs'], row['selected'], round(seconds, 3), flush=True)
                if block == 0 and seed == 17:
                    causal_start = time.perf_counter()
                    changed = p.copy()
                    changed[end:] += 10000
                    other, scale, _ = fit(changed, end, kind, seed)
                    if scale != sd or not all(torch.equal(value, other.state_dict()[name]) for name, value in model.state_dict().items()):
                        raise ValueError('Future perturbation changed fitted parameters or scale')
                    if not np.array_equal(predict(model, sd, p, [end-1]), predict(other, scale, changed, [end-1])):
                        raise ValueError('Future perturbation changed pre-cutoff forecast')
                    causal.append(dict(kind=kind, seed=seed, fit_end=end, passed=True, seconds=time.perf_counter()-causal_start))
                    (R / 'CAUSAL.json').write_text(json.dumps(causal, indent=2), encoding='utf-8')
    complete = len(logs) == 72
    result = dict(complete=complete, fits=len(logs), origins=len(origins), successful_fit_seconds=elapsed_total,
                  causal_refits=len(causal), causal_seconds=sum(row['seconds'] for row in causal),
                  scope='development foundation experiment; existing RKN cell, no algorithm novelty')
    result.update(summarize(saved) if complete else dict(scores={}, development_gate=False))
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
