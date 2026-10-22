# -*- coding: utf-8 -*-
"""Three pre-specified rolling periods for the frozen empirical-tilt candidate.
No modification to frozen source files or permanent CUT configuration.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T
import empirical_bench_round3 as C
import neural_restart as B

P=Path(__file__).resolve().parent
GRID={'ewhs':[32,63,126,252,504,1008], 'recent':[32,64,128,256,512], 'knn':[32,64,128,256,512]}


def metric(y,q,c,p):
    return {'MQS':float(B.score(y,q).mean()),'CRPS':float(c.mean()),
            'WIS':float(B.wis(y,q).mean()),'MAE':float(np.abs(y-q[:,7]).mean()),
            'coverage80':float(np.mean((y>=q[:,2])&(y<=q[:,12]))),
            'coverage95':float(np.mean((y>=q[:,0])&(y<=q[:,14]))),
            'Brier':float(np.mean((p-(np.abs(y)>1e-12))**2))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(B.CUT))
    ap.add_argument('--fractions',nargs='+',type=float,default=[.65,.75,.85])
    ap.add_argument('--window',type=int,default=126)
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    ap.add_argument('--epochs',type=int,default=64)
    ap.add_argument('--tag',default='rolling_tilt_round4');a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    panel=pd.read_csv(B.PANEL,parse_dates=['date']);original_cuts=dict(B.CUT)
    frozen=[P/'empirical_tilt_round3.py',P/'empirical_tilt_round3_followup.py',
            P/'empirical_tilt_round3_followup_predictions.npz']
    before={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in frozen}
    manifest=[]
    for market in a.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        for f in a.fractions:
            ci=int(len(d)*f)
            manifest.append({'market':market,'fraction':f,'cut_index':ci,
                             'cut_date':str(d.date.iloc[ci].date()),
                             'origin_end_exclusive':min(ci+a.window,len(d)-1),
                             'origin_window':a.window})
    report={'status':'exploratory rolling periods; reused historical panel',
            'config':vars(a),'pre_fit_manifest':manifest,
            'primary':'market x fraction x horizon equal-weight geometric MQS; exact CRPS secondary',
            'candidate':'frozen full tilt_rule; five seed probability mixture',
            'memory':'all models fixed pre-cut support; no test-label append',
            'baseline_grids':GRID,'frozen_sha256_before':before,
            'panel_sha256':hashlib.sha256(B.PANEL.read_bytes()).hexdigest(),'periods':{}}
    # Save cuts before observing any losses from these refitted periods.
    (P/(a.tag+'_manifest.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    arrays={};start=time.perf_counter()
    for spec in manifest:
        market=spec['market'];d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        ci=spec['cut_index']
        try:
            B.CUT[market]=spec['cut_date']
            D=T.add_context(d,T.E.build(d))
        finally:B.CUT.clear();B.CUT.update(original_cuts)
        assert D['audit']['cut_index']==ci
        allte=D['te'];D['te']=allte[D['t'][allte]<spec['origin_end_exclusive']]
        te=D['te'];assert len(te)>0
        assert np.all(D['t'][D['rf']]+T.HOR[D['hi'][D['rf']]]<ci)
        assert np.all((D['t'][te]>=ci)&(D['t'][te]<ci+a.window))
        name=f'{market}_f{spec["fraction"]:.2f}'
        print('start',name,spec['cut_date'],'eval_rows',len(te),flush=True)
        pred={};crp={};prob={};fits={};params={}
        q,c,p,f=T.fit(D,'tilt_rule',a.seeds,a.epochs)
        pred['candidate']=q;crp['candidate']=c;prob['candidate']=p;fits['candidate']=f
        print(name,'epochs',[r['best_epoch'] for r in f],flush=True)
        for mode in ['uniform_static','uniform_rule']:
            q,c,p,f=T.fit(D,mode,[11],0)
            pred[mode]=q;crp[mode]=c;prob[mode]=p
        X,bank=C.prepare(D,D['tr']);Xf,bf=C.prepare(D,D['rf'])
        for method,grid in GRID.items():
            trials=[]
            for value in grid:
                q,_,_=C.predict(X,bank,D['va'],method,value,D)
                v=np.mean([B.score(D['z'][D['va']][D['hi'][D['va']]==hi],q[D['hi'][D['va']]==hi]).mean()
                           for hi in range(4)])
                trials.append({'value':value,'normalized_validation_MQS':float(v)})
            chosen=min(trials,key=lambda r:r['normalized_validation_MQS'])
            q,c,p=C.predict(Xf,bf,te,method,chosen['value'],D)
            pred[method]=q*D['scale'][te,None];crp[method]=c*D['scale'][te];prob[method]=p
            params[method]={'selected':chosen,'trials':trials}
        y=D['y'][te];hi=D['hi'][te]
        cells={}
        for hidx,h in enumerate(T.HOR):
            keep=hi==hidx
            cells[str(h)]={'n':int(keep.sum()),'scores':{model:metric(y[keep],q[keep],crp[model][keep],prob[model][keep])
                                                       for model,q in pred.items()}}
        audit=dict(D['audit']);audit['evaluation_rows']=len(te);audit['unrestricted_test_rows']=len(allte)
        audit['evaluation_zero_share']={str(h):float(np.mean(y[hi==i]==0)) for i,h in enumerate(T.HOR)}
        report['periods'][name]={'spec':spec,'audit':audit,'fits':fits,'baseline_selection':params,'by_horizon':cells}
        arrays[name+'_origin']=D['t'][te];arrays[name+'_horizon']=T.HOR[hi];arrays[name+'_target']=y
        arrays[name+'_origin_date']=d.date.to_numpy(dtype='datetime64[D]')[D['t'][te]]
        for model in pred:
            assert np.isfinite(pred[model]).all() and np.min(np.diff(pred[model],axis=1))>=-1e-8
            arrays[name+'_'+model]=pred[model];arrays[name+'_'+model+'_CRPS']=crp[model]
            arrays[name+'_'+model+'_p']=prob[model]
        report['elapsed_seconds']=time.perf_counter()-start
        (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
        print('saved',name,'elapsed_s',round(report['elapsed_seconds'],1),flush=True)
    report['summary']={};report['by_fraction']={}
    def compare(periods):
        summ={}
        for base in ['uniform_static','uniform_rule','ewhs','recent','knn']:
            summ[base]={}
            for m in ['MQS','CRPS']:
                r=[c['scores']['candidate'][m]/c['scores'][base][m] for p in periods for c in p['by_horizon'].values()]
                summ[base][m]={'geometric_gain_percent':float(100*(1-np.exp(np.log(r).mean()))),
                               'wins':int((np.array(r)<1).sum()),'cells':len(r)}
        return summ
    report['summary']=compare(report['periods'].values())
    for frac in a.fractions:
        report['by_fraction'][str(frac)]=compare([p for p in report['periods'].values() if p['spec']['fraction']==frac])
    report['frozen_sha256_after']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in frozen}
    assert before==report['frozen_sha256_after'] and B.CUT==original_cuts
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
