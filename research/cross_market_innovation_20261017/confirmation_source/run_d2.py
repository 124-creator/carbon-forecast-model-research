"""Second frozen development stage: mechanisms, seeds, modern baseline, data size.

D1 remains immutable. Every newly optimized predictor and imputer is counted.
Confirmation files are not available through this runner's data registry.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import traceback
import numpy as np
import torch

from data_core import pack, evaluation_pack, metrics
from fit_core import fit, predict, serialize_state, state_arrays, configure
from run_development import get_markets, with_anchor, dump
from mcssl_control import fit_imputer
from mcssl_fit import fit_mcssl, algebra_check

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent


def config(family, hidden=32, shrink=1., strength=0., lr=.003):
    return dict(id=f'{family}_h{hidden}_s{shrink:g}_l{strength:g}', family=family,
                hidden=hidden, shrink=shrink, strength=strength, lr=lr)


def attach_levels(data, markets):
    levels = np.empty((len(data['origin']),21))
    for mid, market in enumerate(markets):
        selected = data['market'] == mid
        origin = data['origin'][selected]
        levels[selected] = market.price[origin[:,None]+np.arange(-20,1)]
    data['levels'] = levels
    return data


def cap_labeled_windows(data, cap):
    if cap is None:
        return data
    selected = np.concatenate([np.flatnonzero(data['market']==m)[-int(cap):]
                               for m in np.unique(data['market'])])
    out = {k: v[selected] if isinstance(v,np.ndarray) and len(v)==len(data['origin']) else v
           for k,v in data.items()}
    out['supports'] = json.loads(json.dumps(data['supports']))
    for mid,name in enumerate(out['market_names']):
        out['supports'][name]['train'] = out['origin'][out['market']==mid].tolist()
        out['supports'][name]['cap'] = cap
        out['supports'][name]['scaler_uses_common_uncapped_training_history'] = True
    return out


def data_for(markets, task):
    tr = attach_levels(with_anchor(pack(markets,task['cutoff'],'train'),markets),markets)
    va = attach_levels(with_anchor(pack(markets,task['cutoff'],'validation'),markets),markets)
    end = str(int(task['cutoff'][:4])+1)+'-01-01'
    ev = attach_levels(with_anchor(evaluation_pack(markets,task['cutoff'],end,tr['supports']),markets),markets)
    return cap_labeled_windows(tr,task.get('train_cap')), va, ev


def init():
    destination = ROOT/'D2_PRE_RUN.json'
    if destination.exists():
        raise FileExistsError('D2 already frozen')
    names = [m.name for m in get_markets()]
    tasks = []
    def add(c,seed,cutoff,scope='pooled',names_=None,cap=None,purpose='mechanism'):
        tasks.append(dict(config=c,seed=seed,cutoff=cutoff,scope=scope,
                          market_names=names if names_ is None else names_,
                          train_cap=cap,purpose=purpose))
    mechanism = [config('gru'),config('augment'),config('kcm',strength=1.),
                 config('kcm',strength=100.),config('pointwise',strength=1.),
                 config('kcm_unblocked',strength=1.),config('mcssl',strength=1.)]
    for year in (2021,2022,2023):
        cutoff=f'{year}-01-01'
        for seed in (17,29,43):
            for c in mechanism:
                # These exact seed17 configurations already have immutable D1 fits.
                if seed==17 and c['family'] in ('gru','augment'):
                    continue
                if seed==17 and c['family']=='kcm' and c['strength']==1.:
                    continue
                add(c,seed,cutoff)
        for seed in (29,43):
            add(config('gru_price',16,.1),seed,cutoff,purpose='best_D1_baseline_seed')
            add(config('lstm_price',32,1.),seed,cutoff,purpose='best_D1_LSTM_seed')
            for name in names:
                add(config('gru_price',16,1.),seed,cutoff,'local',[name],purpose='best_D1_local_seed')
        for hidden in (16,32):
            for shrink in (.1,1.):
                for name in names:
                    add(config('patchtst',hidden,shrink,lr=.001),17,cutoff,'local',[name],purpose='modern_baseline_selection')
    for cap in (100,400):
        for seed in (17,29,43):
            add(config('gru_price',16,.1),seed,'2023-01-01',cap=cap,purpose='learning_curve')
            add(config('kcm',32,1.,1.),seed,'2023-01-01',cap=cap,purpose='learning_curve')
        for name in names:
            add(config('gru_price',16,1.),17,'2023-01-01','local',[name],cap,purpose='local_learning_curve')
    random.Random(20261018).shuffle(tasks)
    for i,t in enumerate(tasks):
        t['id']=f'd2_{i:03d}'
    hashes={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    plan=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),stage='D2',
              purpose='Development only; full D1 outcomes available when designing D2',
              selection_disclosure='KCM32/shrink1/lambda1 is D1-best KCM; priceGRU16/shrink.1 is D1-best ordinary baseline. Lambda100 is a declared additional development sensitivity, not a pre-D1 hypothesis.',
              formal_predictor_fits=len(tasks),imputer_fits=9,imputer_optimizer_steps_per_fit=500,
              tasks=tasks,maximum_epochs=60,patience=12,deadline_seconds=2400,
              dtype='float64',processes=1,math_threads=1,retry_failed_fits=False,
              train_cap_default=None,validation_origins_per_market=80,purge_horizon=20,
              common_full_training_scaler_for_learning_curve=True,
              primary_score='equal market/year/observation-horizon MSE/RW-MSE at h1,5,20',
              mechanism_comparison='same architecture/initialization/shrink/support, paired seed; KCM against augmentation, pointwise, unblocked, and explicitly adapted MCSSL',
              seed_summary='average squared losses across seeds; seeds are not independent market observations',
              patchtst='unmodified author forecasting core with our short-history/local training protocol; not author benchmark reproduction',
              confirmation_files_allowed=False,source_sha256=hashes,
              registry_sha256=hashlib.sha256((ROOT/'provenance/DEVELOPMENT_REGISTRY.json').read_bytes()).hexdigest(),
              d1_complete_sha256=hashlib.sha256((ROOT/'D1_RUN_COMPLETE.json').read_bytes()).hexdigest())
    dump(destination,plan)
    print(json.dumps(dict(predictor_fits=len(tasks),imputer_fits=9,imputer_steps=4500)),flush=True)


def run():
    from patchtst_fit import fit_patchtst, predict_patchtst
    configure()
    path=ROOT/'D2_PRE_RUN.json';plan=json.loads(path.read_text())
    if (ROOT/'D2_STARTED.json').exists():
        raise FileExistsError('D2 non-resumable; preserve every attempt')
    for name,sha in plan['source_sha256'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('D2 frozen source changed: '+name)
    if hashlib.sha256((ROOT/'provenance/DEVELOPMENT_REGISTRY.json').read_bytes()).hexdigest()!=plan['registry_sha256']:
        raise ValueError('Development registry changed')
    markets=get_markets();by_name={m.name:m for m in markets}
    started=time.monotonic();deadline=started+plan['deadline_seconds']
    dump(ROOT/'D2_STARTED.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),python=platform.python_version(),
         numpy=np.__version__,torch=torch.__version__,pid=os.getpid()))
    receipts=ROOT/'d2_receipts';receipts.mkdir()
    saved=ROOT/'local_only/d2';saved.mkdir(parents=True)
    caches={};imputers={};results=[];failures=[]
    for task in plan['tasks']:
        attempt=dict(task_id=task['id'],started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),task=task)
        with (ROOT/'D2_ATTEMPTS.jsonl').open('a') as f:
            f.write(json.dumps(attempt)+'\n')
        try:
            if time.monotonic()>=deadline:
                raise TimeoutError('Frozen D2 deadline')
            key=(tuple(task['market_names']),task['cutoff'],task['train_cap'])
            if key not in caches:
                caches[key]=data_for([by_name[n] for n in task['market_names']],task)
            tr,va,ev=caches[key];family=task['config']['family']
            if family=='mcssl':
                qkey=(task['cutoff'],task['seed'])
                if qkey not in imputers:
                    qid=f'imputer_{task["cutoff"][:4]}_s{task["seed"]}'
                    dump(receipts/(qid+'_STARTED.json'),dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                         seed=task['seed']+10000,cutoff=task['cutoff'],steps=500,caller_task=task['id']))
                    q,qr=fit_imputer(tr,maxsteps=500,batch_size=256,seed=task['seed']+10000,n_markets=len(tr['market_names']))
                    qr['supports']=tr['supports'];qr['forecaster_seed']=task['seed']
                    dump(receipts/(qid+'.json'),qr)
                    np.savez_compressed(saved/(qid+'_state.npz'),**state_arrays(q))
                    imputers[qkey]=q
                model,receipt=fit_mcssl(tr,va,task['config'],task['seed'],imputers[qkey],plan['maximum_epochs'],deadline)
            elif family=='patchtst':
                model,receipt=fit_patchtst(tr,va,task['config'],task['seed'],max_epochs=plan['maximum_epochs'],deadline=deadline)
            else:
                model,receipt=fit(tr,va,task['config'],task['seed'],max_epochs=plan['maximum_epochs'],deadline=deadline)
            if not receipt.get('eligible_for_selection',True):
                dump(receipts/(task['id']+'_INCOMPLETE.json'),receipt)
                raise TimeoutError('Predictor exhausted deadline; partial fit is ineligible')
            if family=='patchtst':
                forecasts={'full':predict_patchtst(model,ev)}
                states=state_arrays(model['net'])
            else:
                forecasts={v:predict(model,ev,v) for v in ('full','short','sparse')}
                states=serialize_state(model)
            measured={v:metrics(ev,p,v) for v,p in forecasts.items()}
            receipt.update(task=task,supports=tr['supports'],development_metrics=measured,
                           confirmation_data_used=False,fitting_code_sources=plan['source_sha256'])
            dump(receipts/(task['id']+'.json'),receipt)
            np.savez_compressed(saved/(task['id']+'_state.npz'),**states)
            np.savez_compressed(saved/(task['id']+'_predictions.npz'),
                 **{'prediction_'+v:p for v,p in forecasts.items()},raw_y=ev['raw_y'],origin=ev['origin'],
                 dates=ev['dates'],market=ev['market'],scale=ev['scale'],
                 market_names=np.array(ev['market_names']),stat_raw=ev['stat_raw'])
            results.append(dict(task=task,metrics=measured,epochs=receipt['epochs'],
                                selected_epoch=receipt['selected_epoch'],seconds=receipt['seconds']))
            print(json.dumps(dict(done=len(results),total=len(plan['tasks']),id=task['id'],
                 family=family,scope=task['scope'],cutoff=task['cutoff'],cap=task['train_cap'],
                 seed=task['seed'],score=measured['full']['score'],seconds=receipt['seconds'])),flush=True)
        except Exception as exc:
            error=dict(task=task,exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            failures.append(error);dump(receipts/(task['id']+'_FAILED.json'),error)
            print(json.dumps(dict(failed=task['id'],error=str(exc))),flush=True)
            break
    dump(ROOT/'D2_RESULTS.json',dict(results=results))
    dump(ROOT/'D2_RUN_COMPLETE.json',dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         expected_predictor_fits=len(plan['tasks']),completed_predictor_fits=len(results),
         completed_imputer_fits=len(imputers),completed_imputer_steps=500*len(imputers),
         failures=failures,elapsed_seconds=time.monotonic()-started,
         all_completed=len(results)==len(plan['tasks']),confirmation_used=False,additional_retry_fits=0))
    if failures:
        raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['init','run','check'])
    args=parser.parse_args()
    if args.command=='init':init()
    elif args.command=='run':run()
    else:
        result=algebra_check();dump(ROOT/'D2_ALGEBRA_CHECK.json',result);print(json.dumps(result))
