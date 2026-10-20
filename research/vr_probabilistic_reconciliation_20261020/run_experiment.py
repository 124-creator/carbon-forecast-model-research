#!/usr/bin/env python3
"""VR-aware multi-horizon quantile development benchmark.

Reproducible real-data experiment using four user's private input CSVs.
Never save raw data, origin-wise predictions or pricing series to public outputs.
Protocol: see PLAN_BEFORE_FIT_ZH.md. 2022/23 are EXPOSED DEVELOPMENT.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import time
import zlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

BASE = pathlib.Path(__file__).parent
PRIVATE = BASE / 'private_inputs'
W=int(os.environ.get('RECON_WINDOW','252'))
RESULTS = BASE / ('results' if W==252 else 'results_sensitivity_W'+str(W))
RESULTS.mkdir(exist_ok=True)
Q = np.array([.05, .1, .25, .5, .75, .9, .95],dtype=np.float64)
HORIZONS = (5,20)
LAMBDAS = np.array([0, .25, .5, .75, 1.], dtype=np.float64)
SEED = 20261020
K=128
MODELS=('empirical_direct','sqrt_transfer','vr_transfer','fixed_vr_mix','selected_sqrt_mix','selected_vr_mix','joint_20path_bootstrap','coherent_marginal_transport')


def quantile_score(y, pred):
    # Pinball loss across all τ, columns by Q.
    d=np.asarray(y)[...,None]-np.asarray(pred)
    return np.maximum(Q*d,(Q-1)*d).mean(axis=-1)


def qr_from_series(logp, t):
    """At origin t, no historic target has end index >t (includes overlap purge)."""
    samples={}
    for h in HORIZONS:
        ss=np.arange(t-h-W+1,t-h+1,dtype=int)
        assert len(ss)==W and min(ss)>=0 and max(ss+h)<=t
        a=logp[ss+h]-logp[ss]
        assert np.isfinite(a).all()
        samples[h]=a
    s5=samples[5];s20=samples[20]
    mu5,mu20 = float(s5.mean()),float(s20.mean())
    sd5,sd20 = float(s5.std(ddof=1)),float(s20.std(ddof=1))
    ratio = sd20 / max(sd5,1e-10)
    ratio=np.clip(ratio,.5,5.)
    direct={5:np.quantile(s5,Q),20:np.quantile(s20,Q)}
    sqrt_transfer={5:mu5+.5*(direct[20]-mu20),20:mu20+2*(direct[5]-mu5)}
    vr_transfer={5:mu5+1/ratio*(direct[20]-mu20),20:mu20+ratio*(direct[5]-mu5)}
    for d in (direct,sqrt_transfer,vr_transfer):
        for h in HORIZONS:
            assert np.all(np.diff(d[h])>=-1e-12)
    return {'empirical_direct':direct,'sqrt_transfer':sqrt_transfer,
            'vr_transfer':vr_transfer,'fixed_vr_mix':{h:.5*direct[h]+.5*vr_transfer[h] for h in HORIZONS},
            'ratio':ratio, 'mu':{5:mu5,20:mu20}, 'samples':samples}


def fit_lambda(validation):
    """Only validation observations < evaluation start; no test outcome used."""
    selected={}
    for family in ('sqrt','vr'):
        for h in HORIZONS:
            yh=np.array([o['target'][h] for o in validation]);
            empirical=np.stack([o['pred']['empirical_direct'][h] for o in validation])
            transfer=np.stack([o['pred'][family+'_transfer'][h] for o in validation]);
            losses=[]
            for la in LAMBDAS:
                losses.append(float(quantile_score(yh,(1-la)*empirical+la*transfer).mean()))
            # fixed tie breaker prefers less treatment
            best=min(range(len(LAMBDAS)),key=lambda i:(round(losses[i],10),i))
            selected[family+str(h)]={'lambda':float(LAMBDAS[best]),'validation_losses':losses,
                                    'direct_loss':losses[0]}
    return selected


def predicted_for_all(pred, choices):
    direct=pred['empirical_direct'];sf=pred['sqrt_transfer'];vf=pred['vr_transfer']
    return {'empirical_direct':direct,'sqrt_transfer':sf,'vr_transfer':vf,'fixed_vr_mix':pred['fixed_vr_mix'],
       'selected_sqrt_mix':{h:(1-choices['sqrt'+str(h)]['lambda'])*direct[h]+choices['sqrt'+str(h)]['lambda']*sf[h] for h in HORIZONS},
       'selected_vr_mix':{h:(1-choices['vr'+str(h)]['lambda'])*direct[h]+choices['vr'+str(h)]['lambda']*vf[h] for h in HORIZONS}}


def score_energy(scenarios,y):
    # ES(F,y)=mean||X-y||-1/2 E||X-X'||, empirical U-statistic approximation.
    v=np.asarray(scenarios,dtype=float)
    return float(np.linalg.norm(v-np.array(y)[None,:],axis=1).mean() - pdist(v).sum()/len(v)**2)


def rank_map(a, from_sample, target_sorted):
    # Based on midrank of the *same* joint historical forecast sample.
    s=np.sort(from_sample)
    r=(np.searchsorted(s,a,side='left')+np.searchsorted(s,a,side='right'))/2
    u=np.clip(r/W, 0, 1)
    return np.interp(u,np.linspace(0,1,W),target_sorted)


def coherent_scenarios(logp,t,pred,choices,seed):
    start=np.arange(t-20-W+1,t-20+1,dtype=int)
    assert len(start)==W and start.min()>=0 and (start+20).max()<=t
    # paths from entirely observed previous 20-observation intervals.
    d=np.diff(logp)
    windows=np.lib.stride_tricks.sliding_window_view(d,20)
    allpath=windows[start]
    b=allpath.reshape(W,4,5).sum(axis=2)
    joint5=b[:,0];joint20=b.sum(axis=1)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,W,K)
    b0=b[idx].copy()
    # Transfer inverse functions at all 252 empirical probability ranks.
    s5=np.sort(pred['samples'][5]);s20=np.sort(pred['samples'][20]);
    mu5=pred['mu'][5];mu20=pred['mu'][20];rr=pred['ratio']
    lam5=choices['vr5']['lambda'];lam20=choices['vr20']['lambda'];
    target5=(1-lam5)*s5 + lam5*(mu5+(s20-mu20)/rr)
    target20=(1-lam20)*s20 + lam20*(mu20+rr*(s5-mu5))
    z5=rank_map(b0[:,0],joint5,target5)
    z20=rank_map(b0.sum(axis=1),joint20,target20)
    # Exact bridge on four non-overlapping five-step blocks.
    # First segment receives desired 5-step correction; remaining segments
    # receive equal shares of the 20-step remainder; exact sum always holds.
    b1=b0.copy()
    b1[:,0]=z5
    drest=(z20-(b0.sum(axis=1)-b0[:,0])-z5)/3.0
    b1[:,1:]+=drest[:,None]
    assert np.max(np.abs(b1.sum(axis=1)-z20))<1e-12
    assert np.max(np.abs(b1[:,0]-z5))<1e-12
    return (np.column_stack([b0[:,0],b0.sum(axis=1)]),
            np.column_stack([b1[:,0],b1.sum(axis=1)]),
            float(np.max(np.abs(b1.sum(axis=1)-(b1[:,0]+b1[:,1]+b1[:,2]+b1[:,3])))))


def mean_interval_metrics(truth,qmat):
    t=np.asarray(truth);qmat=np.asarray(qmat)
    return {'pinball':float(quantile_score(t,qmat).mean()),
            'coverage80':float(np.mean((t>=qmat[:,1])&(t<=qmat[:,-2]))),
            'coverage90':float(np.mean((t>=qmat[:,0])&(t<=qmat[:,-1]))),
            'width80':float(np.mean(qmat[:,-2]-qmat[:,1])),
            'width90':float(np.mean(qmat[:,-1]-qmat[:,0]))}


def run_cell(market,year, logp, dates,verbose=True):
    b=np.searchsorted(dates,np.datetime64(f'{year}-01-01'))
    e=np.searchsorted(dates,np.datetime64(f'{year+1}-01-01'))
    last=b-20-1
    validation=np.arange(last-79,last+1,dtype=int)
    test=np.arange(b,e-20,dtype=int)
    assert len(validation)==80 and len(test)>=100 and len(set(validation)&set(test))==0
    assert validation[0] >= W+20 and np.max(validation+20)<b
    assert np.max(test+20)<e
    # Simulate origin t before seeing outcome t+h; target attached only after forecast.
    val=[]
    for t in validation:
        prediction=qr_from_series(logp,int(t))
        outcome={h:float(logp[t+h]-logp[t]) for h in HORIZONS}
        val.append({'target':outcome,'pred':prediction})
    choices=fit_lambda(val)
    ys={h:[] for h in HORIZONS}
    stored={m:{h:[] for h in HORIZONS} for m in MODELS}
    esb=[];est=[];c_err=[]
    for t in test:
        prediction=qr_from_series(logp,int(t))
        candidates=predicted_for_all(prediction,choices)
        actual={h:float(logp[t+h]-logp[t]) for h in HORIZONS}
        joint_seed=SEED+100000*year+int(t)+zlib.crc32(market.encode('utf-8'))%10000
        base,transport,coherence_error=coherent_scenarios(logp,int(t),prediction,choices,joint_seed)
        esb.append(score_energy(base,[actual[5],actual[20]]))
        est.append(score_energy(transport,[actual[5],actual[20]]))
        c_err.append(coherence_error)
        candidates['joint_20path_bootstrap']={h:np.quantile(base[:,0 if h==5 else 1],Q) for h in HORIZONS}
        candidates['coherent_marginal_transport']={h:np.quantile(transport[:,0 if h==5 else 1],Q) for h in HORIZONS}
        for h in HORIZONS:
            ys[h].append(actual[h])
            for m in MODELS:stored[m][h].append(candidates[m][h])
    metrics={}
    for m in MODELS:
        metrics[m]={str(h):mean_interval_metrics(ys[h],stored[m][h]) for h in HORIZONS}
        metrics[m]['mean_pinball'] = float(np.mean([metrics[m][str(h)]['pinball'] for h in HORIZONS]))
    metrics['joint_20path_bootstrap']['energy']=float(np.mean(esb))
    metrics['coherent_marginal_transport']['energy']=float(np.mean(est))
    # Private loss arrays used only for paired uncertainty; MUST NOT export.
    losses={m:{h:quantile_score(np.array(ys[h]),np.stack(stored[m][h])) for h in HORIZONS} for m in MODELS}
    rec={'market':market,'year':year,'n_val':len(validation),'n_test':len(test),
         'first_test':str(dates[test[0]])[:10],'last_origin':str(dates[test[-1]])[:10],
         'selection':choices,'metrics':metrics,'coherence_max':float(max(c_err)),
         'strict_asof':True,'exposed_development':True}
    if verbose:print(market,year,'n',len(test),'λ VR',(choices['vr5']['lambda'],choices['vr20']['lambda']),
        'direct',round(metrics['empirical_direct']['mean_pinball'],6),'selectedVR',round(metrics['selected_vr_mix']['mean_pinball'],6),'Δ%',round(100*(metrics['selected_vr_mix']['mean_pinball']/metrics['empirical_direct']['mean_pinball']-1),3),flush=True)
    return rec,losses


def circular_index(rng,n,b):
    starts=rng.integers(0,n,size=int(np.ceil(n/b)))
    return np.concatenate([(s+np.arange(b))%n for s in starts])[:n]


def bootstrap(allloss,nrep=5000):
    # Equal market-year and two-horizon weights, shared resampling for models.
    comparisons=['empirical_direct','fixed_vr_mix','selected_sqrt_mix','vr_transfer','joint_20path_bootstrap']
    selected='selected_vr_mix'
    rng=np.random.default_rng(SEED)
    out={}
    for block in [20,40]:
        gain={m:np.empty(nrep) for m in comparisons}
        for k in range(nrep):
            cell_values={m:[] for m in comparisons}
            for l in allloss:
                n=len(l[selected][5]);ind=circular_index(rng,n,block)
                this=float(np.mean([l[selected][h][ind].mean() for h in HORIZONS]))
                for m in comparisons:
                    ref=float(np.mean([l[m][h][ind].mean() for h in HORIZONS]))
                    cell_values[m].append((this,ref))
            for m in comparisons:
                a,b=np.mean(cell_values[m],axis=0)
                gain[m][k]=100*(1-a/b) if b>0 else np.nan
        out['block'+str(block)]={m:{'lower95_pct':float(np.quantile(gain[m],.025)),
             'upper95_pct':float(np.quantile(gain[m],.975)),
             'p_nonpositive_one_sided':float(np.mean(gain[m]<=0))} for m in comparisons}
    return out


def tests():
    rng=np.random.default_rng(123)
    p=np.r_[2., 2+np.cumsum(np.exp(rng.normal(-3, .4,900)))];
    t=640
    a=qr_from_series(p,t)
    future=p.copy();future[t+1:]+=rng.normal(0,900,size=len(future)-t-1)
    b=qr_from_series(future,t)
    assert np.array_equal(a['empirical_direct'][5],b['empirical_direct'][5])
    assert np.array_equal(a['empirical_direct'][20],b['empirical_direct'][20])
    assert np.array_equal(a['vr_transfer'][5],b['vr_transfer'][5])
    choices={s+str(h):{'lambda':.5} for s in ('sqrt','vr') for h in HORIZONS}
    u,v,e=coherent_scenarios(p,t,a,choices,100)
    assert np.isfinite(u).all() and np.isfinite(v).all() and e<1e-12
    assert np.max(np.abs(np.sort(v[:,0])-np.sort(u[:,0])))>0
    assert np.isfinite(quantile_score(np.array([.2]),np.tile(Q,(1,1)))).all()
    return {'future_perturbation_invariance':True,'coherence_tolerance':e,
            'quantile_noncrossing':True,'shape_transport_changes_distribution':True}


def main():
    t0=time.time(); suite=tests()
    cells=[]; allloss=[];sources={}
    for market in ['GUANGZHOU','HUBEI','SHANGHAI','EUA_LONG']:
        file=PRIVATE/(market+'_development.csv')
        content=file.read_bytes();sources[market]=hashlib.sha256(content).hexdigest()
        df=pd.read_csv(file).sort_values('date')
        dates=pd.to_datetime(df.date).values.astype('datetime64[D]')
        price=np.array(df.price,dtype=float)
        assert (price>0).all() and len(np.unique(dates))==len(dates)
        logp=np.log(price)
        for year in [2022,2023]:
            rec,loss=run_cell(market,year,logp,dates)
            cells.append(rec);allloss.append(loss)
    stats=bootstrap(allloss)
    aggregates={}
    for subset,inds in [('four_market_eight_cell',list(range(8))),('three_chinese_market_six_cell',list(range(6)))]:
        aggregate={}
        for m in MODELS:
            agg={str(h):float(np.mean([cells[i]['metrics'][m][str(h)]['pinball'] for i in inds])) for h in HORIZONS}
            agg['mean_pinball']=float(np.mean([agg[str(h)] for h in HORIZONS]))
            if m in ('joint_20path_bootstrap','coherent_marginal_transport'):
                agg['energy']=float(np.mean([cells[i]['metrics'][m]['energy'] for i in inds]))
            aggregate[m]=agg
        aggregates[subset]=aggregate
    out={'status':'COMPLETE_EXPOSED_DEVELOPMENT_NO_CONFIRMATION','run_date':'2026-10-20','result_identity':('FROZEN_PRIMARY_W252' if W==252 else 'POST_RESULT_SENSITIVITY'),
         'dataset_sha256':sources,'selection':'validation_last80_matured_prior_year_start',
         'max_horizon':20,'window_matured':W,'quantile_grid':Q.tolist(),
         'methods':list(MODELS),'seed':SEED,'scenario_draws':K,
         'synthetic_checks':suite,'cells':cells,'aggregates':aggregates,
         'paired_circular_block_bootstrap':stats,
         'private_data_excluded_from_public_results':True,
         'runtime_s':round(time.time()-t0,3)}
    (RESULTS/'SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    rows=[]
    for c in cells:
        for m in MODELS:
            for h in HORIZONS:
                rows.append({'market':c['market'],'year':c['year'],'model':m,'horizon':h,'n':c['n_test'],
                             **c['metrics'][m][str(h)]})
    pd.DataFrame(rows).to_csv(RESULTS/'SCORES_BY_CELL_HORIZON.csv',index=False)
    agg=aggregates['four_market_eight_cell'];baseline=agg['empirical_direct']['mean_pinball']
    print('\nSUMMARY',[(m,round(v['mean_pinball'],8),round(100*(1-v['mean_pinball']/baseline),3)) for m,v in agg.items()]);
    print('SYNTHETIC',suite,'RUNTIME',round(time.time()-t0,2))
    return out
if __name__=='__main__':main()
