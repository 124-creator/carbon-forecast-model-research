"""Synthetic end-to-end smoke, outcome-access guard, and timing only."""
from __future__ import annotations
import argparse
import copy
from datetime import date, timedelta
import hashlib
import time
import numpy as np
import torch
from data import SPECS, Series, make_case
from run import DEFAULT, NEURAL, add_schedule, scores, write_json, STUDY, utc, baseline_config


def synthetic_case(steps=600, budget=100):
    series = {}
    for i,(name,_,_) in enumerate(SPECS):
        rng = np.random.default_rng(18018+i)
        n=2600
        gaps = rng.choice([1,1,1,1,2], size=n)
        days = np.cumsum(gaps)
        dates = [(date(2015,1,1)+timedelta(days=int(t))).isoformat() for t in days]
        increments=np.zeros(n)
        for t in range(1,n):
            increments[t] = .25*np.tanh(increments[t-1]) + rng.normal(scale=.8+.1*i)
        price=1000+np.cumsum(increments)
        series[name]=Series(name,dates,price)
    return add_schedule(make_case(series,'GUANGZHOU',budget,2022),steps)


class GuardedTest(dict):
    def __getitem__(self,key):
        if key in ('y','delta','base','origin','date','last_label_date'):
            raise AssertionError('A learner attempted to access test outcomes/metadata: '+key)
        return super().__getitem__(key)
    def get(self,key,*args):
        if key in ('y','delta','base','origin','date','last_label_date'):
            raise AssertionError('A learner attempted to access test outcomes/metadata: '+key)
        return super().get(key,*args)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','timing'));a=p.parse_args()
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    torch.set_default_dtype(torch.float64);torch.use_deterministic_algorithms(True)
    from learning import fit
    from baselines import fit_baseline
    from author_kernel import synthetic_check
    cfg=DEFAULT.copy();cfg['max_steps']=12 if a.mode=='smoke' else 600
    case=synthetic_case(cfg['max_steps'],100 if a.mode=='smoke' else 400)
    original_test=case['target_test']
    case['target_test']=GuardedTest(original_test)
    models=NEURAL if a.mode=='smoke' else ['transport']
    evidence={'created_utc':utc(),'mode':a.mode,'market_data_read':False,'source_author_check':synthetic_check(),
              'formal_fits':0,'models':[]}
    for model in models:
        t0=time.perf_counter()
        f=fit_baseline if model.endswith('_gru') else fit
        r=f(case,model,17,baseline_config(cfg) if model.endswith('_gru') else cfg)
        if r.get('receipt',{}).get('eligible') is False:
            raise AssertionError(r['receipt'])
        prediction=np.asarray(r['predictions'])
        val_prediction=np.asarray(r['val_predictions'])
        assert prediction.shape==original_test['y'].shape
        assert val_prediction.shape==case['target_val']['y'].shape
        evaluation=scores(prediction,original_test)
        recomputed=scores(val_prediction,case['target_val'])['score']
        declared=r['receipt'].get('selected_validation_score',r['receipt'].get('validation_score',r['receipt'].get('best_validation_score')))
        if declared is not None and abs(recomputed-declared)>1e-10:
            raise AssertionError(('validation units mismatch',model,recomputed,declared))
        entry={'model':model,'seconds':time.perf_counter()-t0,
               'steps':r['receipt'].get('steps'),'selected_step':r['receipt'].get('selected_step'),
               'test_outcome_guard_pass':True,'finite_forecasts':bool(np.isfinite(prediction).all()),
               'validation_recomputed':recomputed,'validation_declared':declared,
               'receipt':r['receipt']}
        evidence['models'].append(entry)
        print({'model':model,'seconds':round(entry['seconds'],3),'guard':'PASS'},flush=True)
    evidence['pass']=all(x['test_outcome_guard_pass'] and x['finite_forecasts'] for x in evidence['models'])
    write_json(STUDY/f'provenance/SYNTHETIC_{a.mode.upper()}.json',evidence)


if __name__=='__main__':
    main()
