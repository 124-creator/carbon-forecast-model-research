"""Frozen development diagnostic separating recency from labelled-window count.

This stage never changes the already frozen confirmation selection rule.
Recent-tail D2 curves conflate sample amount and historical regime coverage.
The added spread support keeps the complete training history's date range and
common training-only scale; 100 origins are nested in a 400-origin lattice.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'d2_source'))
import argparse
import datetime as dt
import hashlib
import json
import random
import time
import traceback
import numpy as np
from run_development import get_markets,dump
from run_d2 import config,data_for,cap_labeled_windows
from fit_core import fit,predict,serialize_state
from data_core import metrics


def select_spread(data,cap):
    selected=[]
    for mid in np.unique(data['market']):
        allids=np.flatnonzero(data['market']==mid)
        if len(allids)<400:raise ValueError('Need >=400 training origins per market')
        lattice=np.rint(np.linspace(0,len(allids)-1,400)).astype(int)
        use=lattice if cap==400 else lattice[np.rint(np.linspace(0,399,100)).astype(int)]
        selected.extend(allids[use])
    selected=np.array(selected)
    out={k:v[selected] if isinstance(v,np.ndarray) and len(v)==len(data['origin']) else v for k,v in data.items()}
    out['supports']=json.loads(json.dumps(data['supports']))
    for mid,name in enumerate(out['market_names']):
        out['supports'][name]['train']=out['origin'][out['market']==mid].tolist()
        out['supports'][name]['cap']=cap
        out['supports'][name]['support_selection']='uniform temporal lattice;100nestedin400'
        out['supports'][name]['scaler_uses_common_uncapped_training_history']=True
    return out


def init():
    p=ROOT/'LEARNING_DIAGNOSTIC_PRE_RUN.json'
    if p.exists():raise FileExistsError('Learning diagnostic already frozen')
    names=[m.name for m in get_markets()]
    tasks=[]
    for seed in (17,29,43):
        for cap in (100,400):
            for c in [config('gru_price',16,.1),config('kcm',32,1.,1.),config('augment')]:
                tasks.append(dict(config=c,seed=seed,cap=cap,selection='spread',cutoff='2023-01-01',market_names=names))
            tasks.append(dict(config=config('augment'),seed=seed,cap=cap,selection='recent',cutoff='2023-01-01',market_names=names))
    random.Random(20261021).shuffle(tasks)
    for i,t in enumerate(tasks):t['id']=f'ld_{i:03d}'
    plan=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),formal_predictor_fits=len(tasks),
              tasks=tasks,maximum_epochs=60,deadline_seconds=600,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              d2_plan_sha256=hashlib.sha256((ROOT/'D2_PRE_RUN.json').read_bytes()).hexdigest(),
              purpose='Development-only history-length versus label-count diagnostic, motivated by completed D2 learning curve. Not confirmatory evidence.',
              changes_confirmation_selection=False,confirmation_data_allowed=False,
              preprocessing='common uncapped training-only scale; spread400uniform full-history lattice,spread100nested within400;recent supports matched D2 exactly',
              counts_are_labelled_overlapping_windows_not_independent_prices=True,
              fitting_rule='unchanged D2 fit_core,3pairedseeds,fullbatch,float64,thread1,12patience')
    dump(p,plan);print(json.dumps(dict(frozen_fits=len(tasks),confirmation_allowed=False)))


def run():
    plan=json.loads((ROOT/'LEARNING_DIAGNOSTIC_PRE_RUN.json').read_text())
    if (ROOT/'LEARNING_DIAGNOSTIC_STARTED.json').exists():raise FileExistsError('No silent diagnostic reruns')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['source_sha256']:raise ValueError('Source changed')
    d2=json.loads((ROOT/'D2_PRE_RUN.json').read_text())
    for name,sha in d2['source_sha256'].items():
        if hashlib.sha256((ROOT/'d2_source'/name).read_bytes()).hexdigest()!=sha:raise ValueError('D2 source changed')
    started=time.monotonic();deadline=started+plan['deadline_seconds']
    dump(ROOT/'LEARNING_DIAGNOSTIC_STARTED.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    m=get_markets();tr,va,ev=data_for(m,dict(cutoff='2023-01-01',train_cap=None))
    cache={(mode,cap):(select_spread(tr,cap) if mode=='spread' else cap_labeled_windows(tr,cap)) for mode in ('spread','recent') for cap in (100,400)}
    rec=ROOT/'learning_receipts';rec.mkdir();saved=ROOT/'local_only/learning';saved.mkdir(parents=True)
    results=[];failures=[]
    for t in plan['tasks']:
        try:
            chosen=cache[(t['selection'],t['cap'])]
            model,receipt=fit(chosen,va,t['config'],t['seed'],max_epochs=60,deadline=deadline)
            forecast=predict(model,ev)
            score=metrics(ev,forecast)
            receipt.update(task=t,supports=chosen['supports'],development_metrics={'full':score},confirmation_used=False)
            dump(rec/(t['id']+'.json'),receipt)
            np.savez_compressed(saved/(t['id']+'_state.npz'),**serialize_state(model))
            np.savez_compressed(saved/(t['id']+'_predictions.npz'),prediction_full=forecast,raw_y=ev['raw_y'],origin=ev['origin'],dates=ev['dates'],market=ev['market'],scale=ev['scale'],market_names=np.array(ev['market_names']))
            results.append(dict(task=t,metrics={'full':score},epochs=receipt['epochs'],selected_epoch=receipt['selected_epoch'],seconds=receipt['seconds']))
            print(json.dumps(dict(done=len(results),total=len(plan['tasks']),id=t['id'],family=t['config']['family'],selection=t['selection'],cap=t['cap'],score=score['score'])),flush=True)
        except Exception as exc:
            error=dict(task=t,error=str(exc),traceback=traceback.format_exc());failures.append(error);dump(rec/(t['id']+'_FAILED.json'),error);break
    dump(ROOT/'LEARNING_DIAGNOSTIC_RESULTS.json',dict(results=results))
    dump(ROOT/'LEARNING_DIAGNOSTIC_COMPLETE.json',dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         expected_fits=len(plan['tasks']),completed_fits=len(results),all_completed=len(results)==len(plan['tasks']),
         failures=failures,seconds=time.monotonic()-started,confirmation_used=False))
    if failures:raise SystemExit(2)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['init','run']);args=parser.parse_args()
    if args.command=='init':init()
    else:run()
