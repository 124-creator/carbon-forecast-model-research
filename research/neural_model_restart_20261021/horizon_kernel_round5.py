# -*- coding: utf-8 -*-
"""Validation-selected log-horizon reconditioning of a pooled response forest.
The final five-seed forest ensemble is frozen; only five inner seed11 forests
are fitted to select one bandwidth per market, without test selection.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T
import mixed_cdf_forest_round4 as F
import qrf_round4 as Q
import qrf_round4_probability_control as C

P=Path(__file__).resolve().parent
TAUS=[.25,.5,1.,2.,None]


def reweight(w,source_h,query_h,tau):
    if tau is None:factor=np.ones((len(query_h),len(source_h)))
    else:factor=np.exp(-np.abs(np.log(query_h[:,None]/source_h[None,:]))/tau)
    v=w.astype(np.float64)*factor
    v/=v.sum(1,keepdims=True)
    assert np.isfinite(v).all() and np.all(v>=0)
    return v.astype(np.float32)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(T.B.CUT))
    ap.add_argument('--tag',default='horizon_kernel_round5');a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    panel=pd.read_csv(T.B.PANEL,parse_dates=['date'])
    pooled=np.load(P/'mixed_cdf_forest_round4_predictions.npz')
    frozen_paths=[P/'mixed_cdf_forest_round4.py',P/'mixed_cdf_forest_round4_predictions.npz',
                  P/'qrf_round4.py',P/'qrf_round4_probability_control_predictions.npz']
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    report={'status':'exploratory horizon-aware borrowed-label density model','config':vars(a),
            'bandwidth_grid':TAUS,'null_bandwidth':'None=infinity, original pooled model',
            'selection':'one bandwidth per market using inner equal-h normalized MQS; no test selection',
            'inner_forest':dict(F.PARAMS,max_depth=8,random_state=11),
            'final_forest':'frozen five-seed pooled_MSE probabilities, normalized once after horizon weighting',
            'frozen_before':before,'markets':{}}
    arrays={};rows=[];start=time.perf_counter()
    (P/(a.tag+'_manifest.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for market in a.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        D=T.add_context(d,T.E.build(d));key=market+'_';te=D['te'];va=D['va'];tr=D['tr']
        for suffix,actual in [('origin',D['t'][te]),('horizon',T.HOR[D['hi'][te]]),('target',D['y'][te]),('scale',D['scale'][te])]:
            assert np.array_equal(actual,pooled[key+suffix])
            arrays[key+suffix]=actual
        X,_=T.prepare(D,tr,True);x=X['x'].numpy();z=X['z'].numpy()
        forest=RandomForestRegressor(max_depth=8,random_state=11,**F.PARAMS)
        forest.fit(x[tr],z[tr]);w,_=Q.leaf_weights(forest,x[tr],x[va],sanity=True)
        order=np.argsort(z[tr],kind='stable');support=z[tr][order];w=w[:,order]
        source_h=T.HOR[D['hi'][tr[order]]];query_h=T.HOR[D['hi'][va]]
        assert np.all(D['t'][tr]+T.HOR[D['hi'][tr]]<D['audit']['validation_start_index'])
        trials=[]
        for tau in TAUS:
            ww=reweight(w,source_h,query_h,tau)
            (q,_,_),_=Q.ecdf(ww,support,z[va],np.ones(len(va)))
            v=np.mean([T.B.score(D['z'][va][query_h==h],q[query_h==h]).mean() for h in T.HOR])
            trials.append({'tau':tau,'validation_normalized_MQS':float(v)})
        chosen=min(trials,key=lambda r:r['validation_normalized_MQS'])
        cache=np.load(P/f'mixed_cdf_forest_round4_{market}_pooled_MSE_weights.npz')
        ids=cache['pooled_fit_D_indices'];positions=cache['pooled_test_positions']
        assert np.array_equal(positions,np.arange(len(te)))
        assert np.all(D['t'][ids]+T.HOR[D['hi'][ids]]<D['audit']['cut_index'])
        support=cache['pooled_support'];oldw=cache['pooled_ensemble_weights']
        assert np.array_equal(cache['pooled_fit_horizon'],T.HOR[D['hi'][ids]])
        query_h=T.HOR[D['hi'][te]];source_h=cache['pooled_fit_horizon']
        ww=reweight(oldw,source_h,query_h,chosen['tau'])
        (q,c,pr),_=Q.ecdf(ww,support,z[te],D['scale'][te])
        Xf,bf=T.prepare(D,D['rf'],True);p=np.empty(len(te),np.float32)
        for hi,bank in enumerate(bf):
            take=D['hi'][te]==hi
            p[take]=(.8*Xf['local'][te[take]]+.2*torch.full_like(Xf['local'][te[take]],bank['p'])).numpy()
        su,wp,pa=C.replace_probability(support,ww,p)
        qr,cr,_=C.evaluate_distribution(z[te],D['scale'][te],su,wp)
        pred={'horizon_kernel':(q,c,pr),'horizon_kernel_rule':(qr,cr,p)}
        for model,old in [('pooled_original','pooled_MSE_ensemble'),('qrf','qrf_original'),('rule_qrf','qrf_rule_atom'),('lstm_qr','lstm_qr')]:
            pred[model]=(pooled[key+old+'_Q'],pooled[key+old+'_CRPS'] if key+old+'_CRPS' in pooled else None,
                         pooled[key+old+'_p'] if key+old+'_p' in pooled else None)
        borrowed=(ww*(source_h[None,:]!=query_h[:,None])).sum(1,dtype=np.float64)
        cells={}
        for hi,h in enumerate(T.HOR):
            take=query_h==h;y=D['y'][te][take]
            cells[str(h)]={'n':int(take.sum()),'borrowed_mass_mean':float(borrowed[take].mean()),'scores':{}}
            for model,(qq,cc,pp) in pred.items():
                score={'MQS':float(T.B.score(y,qq[take]).mean()),'CRPS':None if cc is None else float(cc[take].mean()),
                       'WIS':float(T.B.wis(y,qq[take]).mean())}
                cells[str(h)]['scores'][model]=score
                rows.append(dict(market=market,horizon=int(h),model=model,**score))
        report['markets'][market]={'selected':chosen,'trials':trials,'audit':D['audit'],'probability_control':pa,'by_horizon':cells}
        for model,(qq,cc,pp) in pred.items():
            arrays[key+model+'_Q']=qq
            if cc is not None:arrays[key+model+'_CRPS']=cc
            if pp is not None:arrays[key+model+'_p']=pp
        arrays[key+'borrowed_mass']=borrowed
        (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        pd.DataFrame(rows).to_csv(P/(a.tag+'_cells.csv'),index=False,encoding='utf-8-sig')
        np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
        print(market,chosen,'borrowed_mean',round(float(borrowed.mean()),4),'elapsed_s',round(time.perf_counter()-start,1),flush=True)
    report['summary']={}
    for model in ['horizon_kernel','horizon_kernel_rule']:
        report['summary'][model]={}
        for base in ['pooled_original','qrf','rule_qrf','lstm_qr']:
            result={}
            for metric in ['MQS','CRPS']:
                rr=[c['scores'][model][metric]/c['scores'][base][metric] for m in report['markets'].values()
                    for c in m['by_horizon'].values() if c['scores'][base][metric] is not None]
                if rr:result[metric]={'geometric_gain_percent':float(100*(1-np.exp(np.log(rr).mean()))),
                                     'wins':int((np.array(rr)<1).sum()),'cells':len(rr)}
            report['summary'][model][base]=result
    report['frozen_after']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    assert before==report['frozen_after']
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
