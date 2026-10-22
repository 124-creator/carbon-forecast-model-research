# -*- coding: utf-8 -*-
"""Matched empirical response-basis ablation; frozen files are read-only."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T
import neural_restart as B

P=Path(__file__).resolve().parent
ORIGINAL_BANK=T.make_bank


def two_factor_bank(D,ids):
    banks=ORIGINAL_BANK(D,ids)
    for bank in banks:
        bank['phi']=bank['phi'].clone()
        bank['phi'][:,2:]=0.
        assert torch.count_nonzero(bank['phi'][:,2:])==0
    return banks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(B.CUT))
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    ap.add_argument('--epochs',type=int,default=64);ap.add_argument('--tag',default='tilt_basis_round4');a=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    panel=pd.read_csv(B.PANEL,parse_dates=['date'])
    cache=np.load(P/'empirical_tilt_round3_followup_predictions.npz')
    frozen=[P/'empirical_tilt_round3.py',P/'empirical_tilt_round3_followup_predictions.npz']
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    report={'status':'exploratory response-basis ablation; reused history',
            'config':vars(a),'change':'same 1407-parameter network; phi columns2:5 zero, preserving sign-location and absolute-amplitude factors',
            'fixed':'same full query context, zero rule, support, CRPS, training maturity filter and refit',
            'full_comparator':'existing frozen full 5seed distribution, not retrained',
            'frozen_before':before,'markets':{}}
    arrays={};start=time.perf_counter()
    T.make_bank=two_factor_bank
    try:
        for market in a.markets:
            d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
            D=T.add_context(d,T.E.build(d));te=D['te'];key=market+'_'
            assert np.array_equal(D['t'][te],cache[key+'origin']) and np.array_equal(D['y'][te],cache[key+'target'])
            q,c,p,f=T.fit(D,'tilt_rule',a.seeds,a.epochs)
            pred={'two_factor':q,'full':cache[key+'full_ensemble_Q'],
                  'uniform_rule':cache[key+'uniform_rule_Q']}
            scores={'two_factor':c,'full':cache[key+'full_ensemble_CRPS'],
                    'uniform_rule':cache[key+'uniform_rule_CRPS']}
            assert np.allclose(p,cache[key+'uniform_rule_p'],atol=5e-7)
            cells={};hi=D['hi'][te];y=D['y'][te]
            for hidx,h in enumerate(T.HOR):
                ix=hi==hidx
                cells[str(h)]={'n':int(ix.sum()),'scores':{name:{'MQS':float(B.score(y[ix],qq[ix]).mean()),
                    'CRPS':float(scores[name][ix].mean())} for name,qq in pred.items()}}
            report['markets'][market]={'fits':f,'audit':D['audit'],'by_horizon':cells}
            arrays[key+'origin']=D['t'][te];arrays[key+'horizon']=T.HOR[hi];arrays[key+'target']=y
            for name in pred:
                arrays[key+name]=pred[name];arrays[key+name+'_CRPS']=scores[name]
            (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
            print(market,'epochs',[r['best_epoch'] for r in f],'elapsed',round(time.perf_counter()-start,1),flush=True)
    finally:T.make_bank=ORIGINAL_BANK
    report['summary']={}
    for name in ['two_factor','uniform_rule']:
        out={}
        for metric in ['MQS','CRPS']:
            rr=np.array([c['scores']['full'][metric]/c['scores'][name][metric]
                        for m in report['markets'].values() for c in m['by_horizon'].values()])
            out[metric]={'geometric_gain_percent':float(100*(1-np.exp(np.log(rr).mean()))),
                         'wins':int((rr<1).sum()),'cells':len(rr)}
        report['summary']['full_vs_'+name]=out
    report['frozen_after']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    assert report['frozen_before']==report['frozen_after']
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
