"""One frozen, finite real-market necessity experiment. No automatic retries."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from data import Data, H, K, HS, TAU, normalize_atoms
from scores import evaluate, crps_numpy


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def distribution_stats(atoms, probs, sigma):
    mean = np.sum(atoms*probs, axis=-1)
    variance = np.sum((atoms-mean[..., None])**2*probs, axis=-1)
    return dict(max_abs_mean=float(np.max(abs(mean))),
                scale_ratio_min=float(np.min(np.sqrt(variance)/sigma)),
                scale_ratio_max=float(np.max(np.sqrt(variance)/sigma)),
                minimum_probability=float(np.min(probs)))


def neural_fit(family, train, val, test, base_atoms, config, private_path, deadline):
    import torch
    from models import Model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.set_default_dtype(torch.float64)
    model = Model(family=family, input_channels=3, H=H, K=K).double()
    if hasattr(model, 'initialize_base'):
        model.initialize_base(torch.tensor(base_atoms))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    sampler = torch.Generator().manual_seed(config['seed'])
    base = torch.tensor(base_atoms)
    tensors = []
    for i, b in enumerate((train, val, test)):
        keys = ('x','sigma') if i == 2 else ('x','sigma','delta')
        tensors.append({k:torch.tensor(b[k]) for k in keys})
    tr, va, te = tensors
    unif_train = np.full((len(train['delta']), H, K), 1/K)
    train_base = np.broadcast_to(base_atoms, unif_train.shape)*train['sigma'][..., None]
    denom = crps_numpy(train_base, unif_train, train['delta']).mean(0)
    val_base = np.broadcast_to(base_atoms, (len(val['delta']), H, K))*val['sigma'][..., None]
    val_denom = crps_numpy(val_base, np.full_like(val_base, 1/K), val['delta']).mean(0)
    if np.any(denom <= 0) or np.any(val_denom <= 0):
        raise ValueError('Degenerate reference score')
    td = torch.tensor(denom)

    def predict(b):
        model.eval()
        aa, pp = [], []
        with torch.no_grad():
            for first in range(0, len(b['x']), config['batch_size']):
                x, sigma = b['x'][first:first+config['batch_size']], b['sigma'][first:first+config['batch_size']]
                a, p = model(x, base[None].expand(len(x),-1,-1))
                aa.append((a*sigma[..., None]).numpy())
                pp.append(p.numpy())
        return np.concatenate(aa), np.concatenate(pp)

    def validation():
        a, p = predict(va)
        return float((crps_numpy(a,p,val['delta']).mean(0)/val_denom)[HS].mean())

    started, t0 = utc(), time.monotonic()
    best_score = validation()
    best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    best_epoch, stale, steps = 0, 0, 0
    trace = [dict(epoch=0, validation_score=best_score, training_loss=None)]
    stop = 'max_epochs'
    for epoch in range(1, config['max_epochs']+1):
        if time.time() >= deadline:
            raise TimeoutError('Frozen cooperative training budget exhausted')
        model.train()
        ids = torch.randperm(len(tr['x']), generator=sampler)
        total, count = 0., 0
        for first in range(0, len(ids), config['batch_size']):
            if time.time() >= deadline:
                raise TimeoutError('Frozen cooperative training budget exhausted')
            idx = ids[first:first+config['batch_size']]
            a, p = model(tr['x'][idx], base[None].expand(len(idx),-1,-1))
            native_a = a*tr['sigma'][idx][..., None]
            loss_cells = model.training_loss(native_a,p,tr['delta'][idx])
            loss = (loss_cells/td).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            steps += 1
            total += float(loss.detach())*len(idx)
            count += len(idx)
        score = validation()
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite validation score')
        trace.append(dict(epoch=epoch, validation_score=score, training_loss=total/count))
        if score < best_score-config['min_delta']:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config['patience']:
            stop = 'patience'
            break
    model.load_state_dict(best_state)
    a, p = predict(te)
    av, pv = predict(va)
    torch.save(best_state, private_path.with_suffix('.pt'))
    info = dict(started_utc=started, completed_utc=utc(), seconds=time.monotonic()-t0,
                fitted=True, seed=config['seed'], sampler_seed=config['seed'], optimizer_steps=steps, completed_epochs=len(trace)-1,
                selected_epoch=best_epoch, validation_score=best_score, stop_reason=stop,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                trace=trace, training_score_denominator=denom.tolist())
    return a,p,av,pv,info


def run_market(spec, protocol, private_dir, deadline):
    from scipy.stats import norm, t
    from statistics import fit_garch, forecast_garch, forecast_garch_fhs
    data = Data(spec, protocol['year'], protocol['validation_size'])
    train = data.batch(data.train, data.stats_end-1)
    val = data.batch(data.val, data.start-1)
    test = data.batch(data.test, data.end-1)
    root = Path(private_dir)/data.name
    root.mkdir()
    rows = []
    fit_cache = {}
    base = data.base_atoms
    normal = normalize_atoms(norm.ppf(TAU))
    t5 = normalize_atoms(t.ppf(TAU, df=5))

    def record(name, a,p,av,pv,info):
        metrics, losses = evaluate(a,p,test)
        validation_metrics, _ = evaluate(av,pv,val)
        np.savez_compressed(root/(name+'.npz'), atoms=a, probabilities=p, truth=test['delta'],
                            sigma=test['sigma'], origin=test['origin'], date=test['date'],
                            validation_atoms=av, validation_probabilities=pv, validation_truth=val['delta'],
                            validation_sigma=val['sigma'], validation_origin=val['origin'], validation_date=val['date'])
        rows.append(dict(market=data.name, family=name, status='complete', metrics=metrics,
                         validation_metrics=validation_metrics, distribution=distribution_stats(a,p,test['sigma']), **info))

    for family in protocol['statistical_families']:
        started, t0 = utc(), time.monotonic()
        try:
            if time.time() >= deadline:
                raise TimeoutError('Frozen deadline reached before statistical job')
            fit = None
            if family.startswith('garch_') and family != 'garch_fhs':
                kind = 'normal' if family == 'garch_normal' else 't5'
                fit = fit_garch(data.price[:data.stats_end], kind=kind)
                fit_cache[family] = fit
            distributions = []
            for batch in (test,val):
                sigma = batch['sigma']
                if family == 'garch_fhs':
                    a = forecast_garch_fhs(data.price, batch['origin'], fit_cache['garch_normal'], draws=2048, seed=1729)
                    distributions.append((a,np.full_like(a,1/K)))
                    continue
                if family == 'fhs_fixed':
                    shape = np.broadcast_to(base, (len(sigma),H,K))
                elif family == 'fhs_rolling':
                    shape = data.rolling_atoms(batch['origin'])
                else:
                    shape = np.broadcast_to(t5 if family.endswith('t5') else normal, (len(sigma),H,K))
                    if fit is not None:
                        sigma = forecast_garch(data.price, batch['origin'], fit)
                a = shape*sigma[...,None]
                distributions.append((a,np.full_like(a,1/K)))
            (a,p),(av,pv) = distributions
            record(family,a,p,av,pv,dict(fitted=fit is not None, fit=fit, started_utc=started,
                   completed_utc=utc(), seconds=time.monotonic()-t0, optimizer_steps=0, completed_epochs=0))
        except Exception as e:
            rows.append(dict(market=data.name, family=family, status='failed', error=repr(e),
                             traceback=traceback.format_exc(), started_utc=started, completed_utc=utc()))
    for family in protocol['neural_families']:
        try:
            a,p,av,pv,info = neural_fit(family,train,val,test,base,protocol['neural'],root/family,deadline)
            record(family,a,p,av,pv,info)
        except Exception as e:
            rows.append(dict(market=data.name, family=family, status='failed', error=repr(e),
                             traceback=traceback.format_exc(), completed_utc=utc()))
    return dict(support=data.receipt(), rows=rows)


def aggregate(results, protocol):
    rows = [r for market in results for r in market['rows']]
    expected = len(protocol['markets'])*(len(protocol['statistical_families'])+len(protocol['neural_families']))
    failed = [r for r in rows if r['status'] != 'complete']
    if failed or len(rows) != expected:
        return dict(status='incomplete', rows=rows, failed=len(failed), expected_jobs=expected, advance=False)
    by = {(r['market'],r['family']):r for r in rows}
    scores, market_scores = {}, {}
    for f in protocol['statistical_families']+protocol['neural_families']:
        cells = {}
        for s in protocol['markets']:
            n = s['name']
            cells[n] = float(np.mean(np.array(by[n,f]['metrics']['crps'])[HS]/np.array(by[n,'fhs_fixed']['metrics']['crps'])[HS]))
        market_scores[f] = cells
        scores[f] = float(np.mean(list(cells.values())))
    gate = {
        'conditional_shape_1pct_better_than_static_and_rolling': scores['conditional_shape'] <= .99*min(scores[x] for x in ('fhs_fixed','fhs_rolling','global_shape')),
        'shape_scale_1pct_better_than_scale_only': scores['shape_scale'] <= .99*scores['scale_only'],
        'shape_scale_better_than_best_statistical': scores['shape_scale'] < min(scores[x] for x in protocol['statistical_families']),
        'conditional_shape_beats_global_in_3_series': sum(market_scores['conditional_shape'][n] < market_scores['global_shape'][n] for n in market_scores['global_shape']) >= 3,
        'shape_scale_beats_scale_only_in_3_series': sum(market_scores['shape_scale'][n] < market_scores['scale_only'][n] for n in market_scores['scale_only']) >= 3,
    }
    return dict(status='complete', rows=rows, expected_jobs=expected, failed=0, scores=scores,
                market_scores=market_scores, gate=gate, advance=all(gate.values()),
                actual_neural_fits=sum(r.get('fitted',False) and r['family'] in protocol['neural_families'] for r in rows),
                actual_statistical_fits=sum(r.get('fitted',False) and r['family'] in protocol['statistical_families'] for r in rows),
                optimizer_steps=sum(r.get('optimizer_steps',0) for r in rows),
                completed_neural_epochs=sum(r.get('completed_epochs',0) for r in rows),
                selected_epoch_zero=sum(r.get('selected_epoch') == 0 for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--private-output', required=True)
    ap.add_argument('--data-map', help='JSON object: the exact five market names to relocated development CSV paths')
    args = ap.parse_args()
    ppath = Path(args.protocol).resolve()
    protocol = json.loads(ppath.read_text())
    study = ppath.parent.parent
    for rel, expected in protocol['source_sha256'].items():
        if hashlib.sha256((study/rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen executable changed: '+rel)
    if args.data_map:
        mapping = json.loads(Path(args.data_map).read_text())
        if not isinstance(mapping,dict) or set(mapping) != {s['name'] for s in protocol['markets']}:
            raise ValueError('The relocation map must contain exactly the five frozen development names')
        for spec in protocol['markets']:
            if not isinstance(mapping[spec['name']], str):
                raise ValueError('A relocated source path must be a string')
            spec['path'] = mapping[spec['name']]
    out, private = Path(args.output),Path(args.private_output)
    out.mkdir(parents=True,exist_ok=False)
    private.mkdir(parents=True,exist_ok=False)
    save(out/'PRE_RUN.json', dict(started_utc=utc(), protocol_sha256=hashlib.sha256(ppath.read_bytes()).hexdigest(),
                                planned_neural_fits=30, planned_statistical_fits=10, planned_forecast_jobs=65,
                                effective_sources=protocol['markets']))
    t0 = time.monotonic()
    deadline = time.time()+protocol['deadline_seconds']
    results = []
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=protocol['cpu_processes'], mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(run_market,s,protocol,str(private),deadline) for s in protocol['markets']]
        for job in as_completed(jobs):
            value = job.result()
            results.append(value)
            save(out/(value['support']['name']+'.json'),value)
            print(json.dumps(dict(market=value['support']['name'],jobs=len(value['rows']),
                                  failures=sum(r['status']!='complete' for r in value['rows'])),ensure_ascii=False),flush=True)
    result = aggregate(results,protocol)
    result.update(completed_utc=utc(),wall_seconds=time.monotonic()-t0,support=[x['support'] for x in results])
    save(out/'RESULTS.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','support','market_scores')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
