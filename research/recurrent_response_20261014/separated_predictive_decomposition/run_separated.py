import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import time
import numpy as np
import torch
from torch import nn
from separated_model import SeparatedForecastModel as ForecastModel

R = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('separated_reference_arrays', R.parent / 'run.py')
if spec is None or spec.loader is None:
    raise RuntimeError('Missing inspected recurrent reference')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
KINDS = ['plain', 'fixed', 'global', 'coupled']
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def fit(prices, end, kind, seed, max_epochs=200):
    ids, x, y, tr, va, scale = base.arrays(prices, end)
    torch.manual_seed(seed)
    model = ForecastModel(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    inputs = torch.tensor(x/scale, dtype=torch.float32)
    targets = torch.tensor(y/scale, dtype=torch.float32)
    with torch.no_grad():
        best = float(((model(inputs[va])-targets[va])**2).mean())
    state = copy.deepcopy(model.state_dict())
    selected, wait, history = 0, 0, []
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss = ((model(inputs[tr])-targets[tr])**2).mean()
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1))
        optimizer.step()
        with torch.no_grad():
            validation = float(((model(inputs[va])-targets[va])**2).mean())
        history.append([float(loss.detach()), validation, gradient_norm])
        if not np.isfinite(history[-1]).all():
            raise ValueError('Nonfinite optimization')
        if validation < best-1e-6:
            best, state = validation, copy.deepcopy(model.state_dict())
            selected, wait = epoch+1, 0
        else:
            wait += 1
        if wait >= 20:
            break
    model.load_state_dict(state)
    diagnostics = {}
    if kind != 'plain':
        with torch.no_grad():
            _, trace = model(inputs[tr[-32:]], return_trace=True)
            rates = trace['rates']
            diagnostics = dict(scope='last 32 training origins and all 60 input positions',
                rate_mean=rates.mean((0, 1)).tolist(), rate_std=rates.std((0, 1)).tolist(),
                reconstruction_max_error=float((trace['components'].sum(-1)-inputs[tr[-32:]]).abs().max()))
            if kind == 'coupled':
                diagnostics['feedback_weight_norm'] = float(model.feedback.weight.norm())
    return model, scale, dict(epochs=len(history), selected=selected, history=history,
        n=len(ids), max_training_target=int(ids[tr[-1]]+20), validation_first=int(ids[va[0]]),
        max_validation_target=int(ids[va[-1]]+20), parameters=sum(p.numel() for p in model.parameters()),
        active_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), diagnostics=diagnostics)


if __name__ == '__main__':
    if (R/'FIT_LOGS.json').exists() or (R/'local_only/PREDICTIONS.npz').exists():
        raise RuntimeError('Existing run preserved; no implicit overwrite or resume')
    paths = [R/'PROTOCOL.md', R/'separated_model.py', Path(__file__), base.R/'run.py', base.R.parent/'torch_segmented_backbone_20261014/prefix_loader.py']
    receipt = dict(status='before_market_fit', source_sha256={str(path.relative_to(R.parents[2])): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (R/'PRE_RUN.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    rows, audit = base.read_prefix(base.R/'../../data/parsed/figshare_28682348/guangzhou_reported.csv', 950, '2021-12-31')
    prices = np.array([float(row['author_reported_price']) for row in rows])
    dates = np.array([row['date'] for row in rows])
    origins = np.array([i for i in range(459, 930) if dates[i][:4] == dates[i+20][:4]])
    if len(origins) != 451:
        raise ValueError('Unexpected evaluation support')
    saved = dict(origins=origins, truth=np.array([prices[i+1:i+21]-prices[i] for i in origins]), years=np.array([d[:4] for d in dates[origins]]))
    (R/'INPUT_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    local = R/'local_only'
    local.mkdir(exist_ok=True)
    logs, causal, elapsed_total = [], [], 0.
    for block in range(12):
        end = 459+40*block
        index = np.where((origins-459)//40 == block)[0]
        for seed in [17, 29]:
            for kind in KINDS:
                if elapsed_total > 1200:
                    break
                started = time.perf_counter()
                model, scale, row = fit(prices, end, kind, seed)
                key = kind+'_'+str(seed)
                saved.setdefault(key, np.full(saved['truth'].shape, np.nan))
                saved[key][index] = base.predictions(model, scale, prices, origins[index])
                seconds = time.perf_counter()-started
                elapsed_total += seconds
                np.savez_compressed(local/(key+'_block'+str(block)+'_parameters.npz'), **{k: v.numpy() for k, v in model.state_dict().items()})
                logs.append(dict(kind=kind, seed=seed, block=block, fit_end=end, seconds=seconds, **row))
                np.savez_compressed(local/'PREDICTIONS.npz', **saved)
                (R/'FIT_LOGS.json').write_text(json.dumps(logs), encoding='utf-8')
                print(kind, seed, block, row['epochs'], row['selected'], round(seconds, 3), flush=True)
                if block == 0 and seed == 17:
                    started = time.perf_counter()
                    changed = prices.copy()
                    changed[end:] += 10000
                    other, other_scale, _ = fit(changed, end, kind, seed)
                    if scale != other_scale or not all(torch.equal(value, other.state_dict()[name]) for name, value in model.state_dict().items()):
                        raise ValueError('Future prices changed fitted parameters or scaler')
                    if not np.array_equal(base.predictions(model, scale, prices, [end-1]), base.predictions(other, other_scale, changed, [end-1])):
                        raise ValueError('Future prices changed pre-cutoff forecast')
                    causal.append(dict(kind=kind, seed=seed, fit_end=end, passed=True, seconds=time.perf_counter()-started))
                    (R/'CAUSAL.json').write_text(json.dumps(causal, indent=2), encoding='utf-8')
    complete = len(logs) == 96
    scores = {}
    if complete:
        for kind in KINDS:
            cells, by_seed = [], {}
            for year in ['2020', '2021']:
                mask = saved['years'] == year
                for h in [1, 5, 20]:
                    denominator = np.mean(saved['truth'][mask, h-1]**2)
                    numerator = np.mean([np.mean((saved[kind+'_'+str(seed)][mask, h-1]-saved['truth'][mask, h-1])**2) for seed in [17, 29]])
                    cells.append(dict(year=year, h=h, ratio=float(numerator/denominator)))
            for seed in [17, 29]:
                by_seed[str(seed)] = float(np.mean([np.mean((saved[kind+'_'+str(seed)][saved['years']==year, h-1]-saved['truth'][saved['years']==year, h-1])**2)/np.mean(saved['truth'][saved['years']==year, h-1]**2) for year in ['2020', '2021'] for h in [1, 5, 20]]))
            scores[kind] = dict(score=float(np.mean([cell['ratio'] for cell in cells])), cells=cells, seed_scores=by_seed,
                year_scores={year: float(np.mean([cell['ratio'] for cell in cells if cell['year']==year])) for year in ['2020', '2021']})
    mechanism = complete and all(scores['coupled']['score'] < scores[kind]['score'] for kind in ['plain', 'fixed', 'global']) and all(scores['coupled']['year_scores'][year] <= scores['global']['year_scores'][year] for year in ['2020', '2021'])
    result = dict(complete=complete, fits=len(logs), origins=len(origins), successful_fit_seconds=elapsed_total,
        causal_refits=len(causal), causal_seconds=sum(row['seconds'] for row in causal), scores=scores,
        mechanism_gate=bool(mechanism), direction_escalation_gate=bool(mechanism and scores['coupled']['score'] < .9497932301207621),
        scope='full model development hypothesis; novelty and publication not established')
    (R/'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
