# -*- coding: utf-8 -*-
"""Strong empirical-distribution controls for the neural tail-tilting candidate.
All hyperparameters use inner validation; refit support contains only pre-cut
matured labels. No forecaster in this comparison appends test labels to memory.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T
import neural_restart as B

P=Path(__file__).resolve().parent


def prepare(D,ids):
    X,banks=T.prepare(D,ids,True)
    for hi,bank in enumerate(banks):
        ii=ids[(D['hi'][ids]==hi)&(np.abs(D['y'][ids])>1e-12)]
        ii=ii[np.argsort(D['z'][ii],kind='stable')]
        assert np.array_equal(D['t'][ii]+T.HOR[hi],bank['mature'].numpy())
        bank['keys']=X['x'][ii]
    return X,banks


def weights(X,idx,bank,method,param):
    n=len(idx);m=len(bank['z'])
    if method=='ewhs':
        # Current query origin contributes the same common factor, which cancels
        # when a fixed pre-cut memory is normalized. This is a frozen EWHS.
        age=bank['mature'].max()-bank['mature']
        logw=-np.log(2.)*age.float()/param
        mark=logw.softmax(0)[None,:].expand(n,-1)
    elif method=='recent':
        keep=torch.argsort(bank['mature'],descending=True)[:min(int(param),m)]
        v=torch.zeros(m);v[keep]=1/len(keep)
        mark=v[None,:].expand(n,-1)
    elif method=='knn':
        dis=torch.cdist(X['x'][idx],bank['keys'])
        nearest=dis.topk(min(int(param),m),largest=False,sorted=False).indices
        mark=torch.zeros((n,m));mark.scatter_(1,nearest,1/nearest.shape[1])
    else:raise ValueError(method)
    p=.8*X['local'][idx]+.2*bank['p']
    j=bank['zero_pos']
    return torch.cat([p[:,None]*mark[:,:j],(1-p)[:,None],p[:,None]*mark[:,j:]],1)


def predict(X,banks,ids,method,param,D):
    ids=torch.tensor(ids);Q=np.empty((len(ids),len(T.TT)));S=np.empty(len(ids));Pr=np.empty(len(ids))
    for hi,bank in enumerate(banks):
        keep=np.flatnonzero(D['hi'][ids.numpy()]==hi);ix=ids[keep]
        if not len(ix):continue
        with torch.no_grad():
            w=weights(X,ix,bank,method,param)
            q=T.quantiles(w,bank['support']).numpy()
            s=T.crps(X['z'][ix],w,bank['support']).numpy()
        Q[keep]=q;S[keep]=s;Pr[keep]=1-w[:,bank['zero_pos']].numpy()
    return Q,S,Pr


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(B.CUT))
    ap.add_argument('--tag',default='empirical_bench_round3');a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    panel=pd.read_csv(B.PANEL,parse_dates=['date'])
    old=np.load(P/'empirical_tilt_round3_predictions.npz')
    meta=json.loads((P/'empirical_tilt_round3.json').read_text(encoding='utf-8'))
    assert meta['panel_sha256']==hashlib.sha256(B.PANEL.read_bytes()).hexdigest()
    report={'status':'exploratory baseline comparison; historically reused tests',
            'memory':'fixed at each fit boundary; no test-label append',
            'zero_mass':'same .8 local + .2 static nonzero rule as tilt_rule',
            'selection':'inner equal-horizon normalized pinball',
            'candidates':{'ewhs':[32,63,126,252,504,1008],'recent':[32,64,128,256,512],
                          'knn':[32,64,128,256,512]},
            'panel_sha256':meta['panel_sha256'],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'markets':{}}
    arrays={};start=time.perf_counter()
    for market in a.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        D=T.add_context(d,T.E.build(d));X,banks=prepare(D,D['tr']);Xf,bf=prepare(D,D['rf'])
        te=D['te'];hi=D['hi'][te];y=D['y'][te]
        assert np.array_equal(D['t'][te],old[market+'_origin']) and np.array_equal(y,old[market+'_target'])
        result={'selected':{},'validation_trials':{},'by_horizon':{}}
        pred={name:old[market+'_'+name] for name in ['empirical_constant','uniform_rule','tilt_rule','tilt_neural']}
        crp={name:old[market+'_'+name+'_CRPS'] for name in pred}
        for method,params in report['candidates'].items():
            trials=[];best=None
            for param in params:
                q,s,p=predict(X,banks,D['va'],method,param,D)
                v=float(np.mean([B.score(D['z'][D['va']][D['hi'][D['va']]==h],q[D['hi'][D['va']]==h]).mean()
                                 for h in range(len(T.HOR))]))
                trials.append({'param':param,'validation_normalized_MQS':v})
                if best is None or v<best['validation_normalized_MQS']-1e-9:best=trials[-1]
            q,s,p=predict(Xf,bf,te,method,best['param'],D)
            pred[method]=q*D['scale'][te,None];crp[method]=s*D['scale'][te]
            result['selected'][method]=best;result['validation_trials'][method]=trials
            print(market,method,best,flush=True)
        for hidx,h in enumerate(T.HOR):
            keep=hi==hidx
            result['by_horizon'][str(h)]={'n':int(keep.sum()),'scores':{
               name:{'MQS':float(B.score(y[keep],q[keep]).mean()),'CRPS':float(crp[name][keep].mean()),
                     'WIS':float(B.wis(y[keep],q[keep]).mean()),
                     'coverage80':float(np.mean((y[keep]>=q[keep,2])&(y[keep]<=q[keep,12]))),
                     'coverage95':float(np.mean((y[keep]>=q[keep,0])&(y[keep]<=q[keep,14])))}
               for name,q in pred.items()}}
        report['markets'][market]=result
        arrays[market+'_origin']=D['t'][te];arrays[market+'_horizon']=T.HOR[hi];arrays[market+'_target']=y
        for name,q in pred.items():arrays[market+'_'+name]=q;arrays[market+'_'+name+'_CRPS']=crp[name]
        (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
    report['summary']={}
    for base in ['empirical_constant','uniform_rule','ewhs','recent','knn','tilt_neural']:
        out={}
        for metric in ['MQS','CRPS']:
            r=[c['scores']['tilt_rule'][metric]/c['scores'][base][metric]
               for m in report['markets'].values() for c in m['by_horizon'].values()]
            out[metric]={'geometric_gain_percent':float(100*(1-np.exp(np.log(r).mean()))),
                         'wins':int((np.array(r)<1).sum()),'cells':len(r)}
        report['summary']['tilt_rule_vs_'+base]=out
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
