"""Single preregistered temporal confirmation: no test-driven configuration.

Only run() opens the sealed Shanghai prices, after checking the frozen complete
task list, source and selection manifest. Configurations never inspect scores.
The 2025 refresh may use matured 2024 observations by the frozen calendar rule.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import csv
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

from data_core import Market, pack, evaluation_pack, metrics
from fit_core import configure, fit, predict, serialize_state, state_arrays
from run_development import get_markets, with_anchor, dump
from run_d2 import config, attach_levels
from mcssl_control import fit_imputer
from mcssl_fit import fit_mcssl
from patchtst_fit import fit_patchtst, predict_patchtst

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).resolve().parent
DEFAULT_SEALED=Path('/workspace/scratch/b31c5edbd0de/scratch/data_expansion_20261017/local_only/confirmation_sealed/SHANGHAI_D4.csv')


def init():
    destination=ROOT/'CONFIRMATION_PRE_RUN.json'
    if destination.exists():
        raise FileExistsError('Confirmation plan already frozen')
    rules=json.loads((ROOT/'CONFIRMATION_DESIGN_RULES.json').read_text())
    selection=json.loads((ROOT/'DEVELOPMENT_AGGREGATE.json').read_text())
    if not json.loads((ROOT/'D2_RUN_COMPLETE.json').read_text())['all_completed']:
        raise ValueError('Development is incomplete')
    names=[m.name for m in get_markets()]
    shared=[selection['selected_candidate']['config'],config('augment'),config('gru'),
            config('pointwise',strength=1.),config('kcm_unblocked',strength=1.),
            config('mcssl',strength=1.),config('gru_price',16,.1),config('lstm_price',32,1.)]
    tasks=[]
    for cutoff in rules['calendar_training_cutoffs']:
        for seed in rules['final_seeds']:
            for c in shared:
                tasks.append(dict(config=c,seed=seed,cutoff=cutoff,scope='pooled',market_names=names))
            tasks.append(dict(config=config('gru_price',16,1.),seed=seed,cutoff=cutoff,scope='local',market_names=['SHANGHAI_D4']))
            tasks.append(dict(config=selection['selected_patchtst']['config'],seed=seed,cutoff=cutoff,scope='local',market_names=['SHANGHAI_D4']))
        tasks.append(dict(config=selection['selected_ridge']['config'],seed=17,cutoff=cutoff,scope='pooled',market_names=names))
    random.Random(20261020).shuffle(tasks)
    for i,t in enumerate(tasks):t['id']=f'c_{i:03d}'
    hashes={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    plan=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),stage='C',
              formal_predictor_fits=len(tasks),imputer_fits=6,imputer_steps_per_fit=500,
              tasks=tasks,maximum_epochs=60,patience=12,deadline_seconds=1800,
              math_threads=1,processes=1,dtype='float64',retry_failed_fits=False,
              source_sha256=hashes,sealed_path=str(DEFAULT_SEALED),
              sealed_sha256=rules['confirmation_source_sha256'],
              data_registry_sha256=hashlib.sha256((ROOT/'provenance/DEVELOPMENT_REGISTRY.json').read_bytes()).hexdigest(),
              rules_sha256=hashlib.sha256((ROOT/'CONFIRMATION_DESIGN_RULES.json').read_bytes()).hexdigest(),
              selection_sha256=hashlib.sha256((ROOT/'DEVELOPMENT_AGGREGATE.json').read_bytes()).hexdigest(),
              selected_candidate=selection['selected_candidate']['config'],
              selected_baseline={k:selection['selected_best_baseline'][k] for k in ('config','scope','config_id','score')},
              prices_opened_before_freeze=False,confirmation_scored_before_freeze=False,
              observation_horizons=[1,5,20],complete_same_calendar_year_H20=True,
              final_refits_use_all_mature_training_history_except_fixed80validation_and20purge=True,
              comparison_note='Three forecast seeds. Ridge/RW/stat deterministic. Every selected baseline and fixed mechanism arm is scored; no per-test-cell selection.')
    dump(destination,plan)
    print(json.dumps(dict(frozen_confirmation_predictor_fits=len(tasks),imputer_fits=6,prices_read=False)),flush=True)


def open_markets(plan):
    markets=get_markets()
    path=Path(plan['sealed_path']);blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=plan['sealed_sha256']:
        raise ValueError('Sealed Shanghai data changed')
    rows=list(csv.DictReader(blob.decode('utf-8-sig').splitlines()))
    dates=np.array([r['date'] for r in rows]);price=np.array([float(r['price']) for r in rows])
    if len(rows)!=261 or np.any(dates[1:]<=dates[:-1]) or np.any(dates<'2024-01-01') or np.any(dates>='2026-01-01'):
        raise ValueError('Unexpected sealed date bounds/count')
    if not np.isfinite(price).all() or np.any(price<=0):
        raise ValueError('Invalid sealed price')
    for i,m in enumerate(markets):
        if m.name=='SHANGHAI_D4':
            assert m.dates[-1]<dates[0]
            days=np.array([dt.date.fromisoformat(str(d)).toordinal() for d in dates])
            combined=hashlib.sha256((m.source_sha256+plan['sealed_sha256']).encode()).hexdigest()
            markets[i]=Market(m.name,np.r_[m.dates,dates],np.r_[m.day,days],np.r_[m.price,price],combined,m.source_path+' + sealed Shanghai')
    return markets


def data_for(markets,task):
    chosen=[m for name in task['market_names'] for m in markets if m.name==name]
    train=attach_levels(with_anchor(pack(chosen,task['cutoff'],'train'),chosen),chosen)
    val=attach_levels(with_anchor(pack(chosen,task['cutoff'],'validation'),chosen),chosen)
    sh=[m for m in chosen if m.name=='SHANGHAI_D4']
    end=str(int(task['cutoff'][:4])+1)+'-01-01'
    ev=attach_levels(with_anchor(evaluation_pack(sh,task['cutoff'],end,train['supports']),sh),sh)
    ev['market_names']=[m.name for m in chosen]
    ev['market']=np.full(len(ev['origin']),ev['market_names'].index('SHANGHAI_D4'),dtype=np.int64)
    return train,val,ev


def run():
    configure();path=ROOT/'CONFIRMATION_PRE_RUN.json';plan=json.loads(path.read_text())
    if (ROOT/'CONFIRMATION_STARTED.json').exists():
        raise FileExistsError('Confirmation non-resumable; no silent test reuse')
    for name,sha in plan['source_sha256'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen confirmation source changed: '+name)
    for name,key in [('CONFIRMATION_DESIGN_RULES.json','rules_sha256'),('DEVELOPMENT_AGGREGATE.json','selection_sha256'),('provenance/DEVELOPMENT_REGISTRY.json','data_registry_sha256')]:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=plan[key]:
            raise ValueError('Frozen confirmation dependency changed: '+name)
    started=time.monotonic();deadline=started+plan['deadline_seconds']
    dump(ROOT/'CONFIRMATION_STARTED.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),python=platform.python_version(),
         numpy=np.__version__,torch=torch.__version__,pid=os.getpid(),first_root_price_access_follows_this_record=True))
    markets=open_markets(plan)
    receipts=ROOT/'confirmation_receipts';receipts.mkdir()
    saved=ROOT/'local_only/confirmation';saved.mkdir(parents=True)
    caches={};imputers={};results=[];failures=[];controls={}
    for task in plan['tasks']:
        with (ROOT/'CONFIRMATION_ATTEMPTS.jsonl').open('a') as f:
            f.write(json.dumps(dict(task_id=task['id'],utc=dt.datetime.now(dt.timezone.utc).isoformat(),task=task))+'\n')
        try:
            if time.monotonic()>=deadline:raise TimeoutError('Frozen confirmation deadline')
            key=(tuple(task['market_names']),task['cutoff'])
            if key not in caches:caches[key]=data_for(markets,task)
            tr,va,ev=caches[key];family=task['config']['family']
            if task['cutoff'] not in controls:
                controls[task['cutoff']]=dict(rw=metrics(ev,np.zeros_like(ev['raw_y'])),stat=metrics(ev,ev['stat_raw']))
            if family=='mcssl':
                qkey=(task['cutoff'],task['seed'])
                if qkey not in imputers:
                    qid=f'imputer_{task["cutoff"][:4]}_s{task["seed"]}'
                    dump(receipts/(qid+'_STARTED.json'),dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),steps=500,caller_task=task['id']))
                    q,qr=fit_imputer(tr,maxsteps=500,batch_size=256,seed=task['seed']+10000,n_markets=len(tr['market_names']))
                    qr.update(supports=tr['supports'],forecaster_seed=task['seed'])
                    dump(receipts/(qid+'.json'),qr)
                    np.savez_compressed(saved/(qid+'_state.npz'),**state_arrays(q));imputers[qkey]=q
                model,receipt=fit_mcssl(tr,va,task['config'],task['seed'],imputers[qkey],plan['maximum_epochs'],deadline)
            elif family=='patchtst':
                model,receipt=fit_patchtst(tr,va,task['config'],task['seed'],max_epochs=plan['maximum_epochs'],deadline=deadline)
            else:
                model,receipt=fit(tr,va,task['config'],task['seed'],max_epochs=plan['maximum_epochs'],deadline=deadline)
            if not receipt.get('eligible_for_selection',True):
                dump(receipts/(task['id']+'_INCOMPLETE.json'),receipt)
                raise TimeoutError('Incomplete predictor is not eligible')
            if family=='patchtst':
                forecasts={'full':predict_patchtst(model,ev)};states=state_arrays(model['net'])
            else:
                forecasts={v:predict(model,ev,v) for v in ('full','short','sparse')};states=serialize_state(model)
            measured={v:metrics(ev,p,v) for v,p in forecasts.items()}
            receipt.update(task=task,supports=tr['supports'],confirmation_metrics=measured,
                           hyperparameters_selected_by_development_only=True,other_market_prices_end_before_2024=True)
            dump(receipts/(task['id']+'.json'),receipt)
            np.savez_compressed(saved/(task['id']+'_state.npz'),**states)
            np.savez_compressed(saved/(task['id']+'_predictions.npz'),
                 **{'prediction_'+v:p for v,p in forecasts.items()},raw_y=ev['raw_y'],origin=ev['origin'],dates=ev['dates'],
                 market=ev['market'],scale=ev['scale'],market_names=np.array(ev['market_names']),stat_raw=ev['stat_raw'])
            results.append(dict(task=task,metrics=measured,epochs=receipt['epochs'],selected_epoch=receipt['selected_epoch'],seconds=receipt['seconds']))
            # No test score is printed until the entire frozen run finishes.
            print(json.dumps(dict(done=len(results),total=len(plan['tasks']),id=task['id'],family=family,seed=task['seed'],cutoff=task['cutoff'],seconds=receipt['seconds'])),flush=True)
        except Exception as exc:
            error=dict(task=task,exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            failures.append(error);dump(receipts/(task['id']+'_FAILED.json'),error)
            print(json.dumps(dict(failed=task['id'],error=str(exc))),flush=True);break
    dump(ROOT/'CONFIRMATION_RESULTS.json',dict(results=results,controls=controls))
    dump(ROOT/'CONFIRMATION_RUN_COMPLETE.json',dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         expected_predictor_fits=len(plan['tasks']),completed_predictor_fits=len(results),
         completed_imputer_fits=len(imputers),completed_imputer_steps=500*len(imputers),
         failures=failures,all_completed=len(results)==len(plan['tasks']),elapsed_seconds=time.monotonic()-started,
         additional_retry_fits=0,confirmation_scores_used_for_configuration_changes=False))
    if failures:raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['init','run'])
    args=parser.parse_args()
    if args.command=='init':init()
    else:run()
