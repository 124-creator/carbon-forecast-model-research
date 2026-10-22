"""Frozen estimand and descriptive, paired dependence-aware uncertainty."""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
from experiment import jwrite
import core

def losses(pred,truth):return ((pred-truth[None])**2).mean(0)
def score(l,truth,years):
    return float(np.mean([l[years==y,h-1].mean()/np.mean(truth[years==y,h-1]**2) for y in ('2020','2021') for h in (1,5,20)]))
def block_intervals(p,l,truth,years,comparisons,rep=2000):
    result={c:{} for c in comparisons};rng=np.random.default_rng(7701)
    for length in (20,40,60):
        own=np.zeros(rep);other={c:np.zeros(rep) for c in comparisons}
        for year in ('2020','2021'):
            subset=np.where(years==year)[0];n=len(subset);starts=rng.integers(0,n-length+1,size=(rep,int(np.ceil(n/length))))
            idx=subset[(starts[...,None]+np.arange(length)).reshape(rep,-1)[:,:n]]
            for h in (1,5,20):
                den=np.mean(truth[idx,h-1]**2,axis=1)
                own+=np.mean(l['OVERLAP'][idx,h-1],axis=1)/den/6
                for c in comparisons:other[c]+=np.mean(l[c][idx,h-1],axis=1)/den/6
        for c in comparisons:result[c][str(length)]=np.percentile(100*(1-own/other[c]),[2.5,97.5]).tolist()
    return result

def run(out):
    out=Path(out)
    with np.load(out/'evaluation.npz',allow_pickle=False) as z:
        valid=z['valid'];truth=z['truth'][valid];years=z['years'][valid];orig=z['origins'][valid]
    preds={f.stem:np.load(f,allow_pickle=False)['prediction'][:,valid] for f in sorted((out/'predictions').glob('*.npz'))}
    rows=[];cells=[];l={};ss={}
    for name,pr in preds.items():
        if not np.isfinite(pr[:,:,[0,4,19]]).all():raise ValueError(f'Missing scored prediction: {name}')
        l[name]=losses(pr,truth);mean=score(l[name],truth,years);ens=score(losses(pr.mean(0,keepdims=True),truth),truth,years)
        ss[name]=[score(losses(pr[s:s+1],truth),truth,years) for s in range(len(pr))]
        rows.append(dict(name=name,mean_seed_relative_MSE=mean,prediction_ensemble_relative_MSE=ens,seed_count=len(pr)))
        for year in ('2020','2021'):
            mask=years==year
            for h in (1,5,20):
                err=pr[:,mask,h-1]-truth[mask,h-1][None]
                cells.append(dict(name=name,year=year,h=h,n=int(mask.sum()),relative_MSE=float(np.mean(err**2)/np.mean(truth[mask,h-1]**2)),
                    mean_seed_RMSE=float(np.sqrt(np.mean(err**2,axis=1)).mean()),mean_seed_MAE=float(np.abs(err).mean()),ensemble_RMSE=float(np.sqrt(np.mean(err.mean(0)**2)))))
    df=pd.DataFrame(rows).sort_values('mean_seed_relative_MSE');df.to_csv(out/'all_scores.csv',index=False);pd.DataFrame(cells).to_csv(out/'cells.csv',index=False)
    scores=df.set_index('name')['mean_seed_relative_MSE'].to_dict();ensembles=df.set_index('name')['prediction_ensemble_relative_MSE'].to_dict()
    pool=[k for k in scores if k!='OVERLAP'];star=min(pool,key=scores.get);estar=min(pool,key=ensembles.get)
    gains={k:1-scores['OVERLAP']/scores[k] for k in pool};wins={k:int(np.sum(np.array(ss['OVERLAP'])<np.array(ss[k]))) for k in core.MODES if k!='OVERLAP'}
    annual={y:float(np.mean([c['relative_MSE'] for c in cells if c['name']=='OVERLAP' and c['year']==y])) for y in ('2020','2021')}
    mechanism=all(gains[k]>=.01 for k in ('SOFTMAX','DIAGONAL','THIN','SHUFFLED')) and wins['DIAGONAL']>=3
    accuracy=(scores['OVERLAP']<=.98 and gains[star]>=.01 and ensembles['OVERLAP']<ensembles[estar] and max(annual.values())<=1.05)
    comparisons=list(dict.fromkeys(['RW','DIAGONAL','SOFTMAX','THIN','SHUFFLED','FULL',star]))
    intervals=block_intervals(preds,l,truth,years,comparisons)
    fits=json.loads((out/'FIT_INDEX.json').read_text());account={}
    for mode in core.MODES:
        f=[a for a in fits if a['mode']==mode]
        account[mode]=dict(fits=len(f),epochs=sum(a['epochs'] for a in f),initialization_selected=sum(a['selected_epoch']==0 for a in f),sum_fit_seconds=sum(a['seconds'] for a in f),allocated_parameters=f[0]['allocated_parameters'])
        if 'weight_diagnostics' in f[0]:account[mode]['weight_diagnostics']={k:float(np.mean([a['weight_diagnostics'][k] for a in f])) for k in f[0]['weight_diagnostics']}
    result=dict(status='PASS_DEVELOPMENT_ONLY' if mechanism and accuracy else 'STOP_DEVELOPMENT_GATE_FAILED',source_commit=core.BASE_SHA,protocol_commit=core.PROTOCOL_SHA,
       candidate='OVERLAP',main_score=scores['OVERLAP'],ensemble_score=ensembles['OVERLAP'],scores=scores,ensembles=ensembles,seed_scores=ss,Cstar_mean=star,Cstar_ensemble=estar,
       gains=gains,paired_seed_wins=wins,annual=annual,mechanism_pass=bool(mechanism),accuracy_pass=bool(accuracy),origins=len(orig),configurations=len(preds),
       bootstrap_gain_percent_95=intervals,fit_accounting=account,uncertainty_scope='Repeatedly used development period. No selection or multiplicity correction; not confirmatory.',
       new_history_read=False,new_market_read=False,novelty_established=False,raw_observation_dates_not_verified_trading_calendar=True)
    jwrite(out/'RESULTS.json',result);print(df.to_string(index=False));print(result['status'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
