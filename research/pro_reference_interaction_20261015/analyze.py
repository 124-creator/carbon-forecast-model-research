"""Fixed development estimand. No model fitting or holdout access."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from model import load_prefix,HORIZONS,SEEDS
from experiment import dump

def panel(data_path,out):
    p,d,receipt=load_prefix(data_path)
    origins=np.array([t for t in range(459,930) if d[t][:4]==d[t+20][:4]])
    if len(origins)!=451:raise ValueError('Scoring population changed')
    years=np.array([int(d[t][:4]) for t in origins])
    truth=np.array([[p[t+h]-p[t] for h in HORIZONS] for t in origins])
    pred={}
    for path in sorted((out/'predictions').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:a=z['pred'][:,origins,:][:,:,[h-1 for h in HORIZONS]]
        if not np.isfinite(a).all():raise ValueError(f'Incomplete {path.name}')
        pred[path.stem]=a
    return origins,years,truth,pred

def scores(a,truth,years):
    each=[];cells=[]
    for year in (2020,2021):
        idx=years==year;denom=np.mean(truth[idx]**2,axis=0)
        if np.any(denom<=0):raise ValueError('Zero RW error')
        ratios=np.mean((a[:,idx]-truth[None,idx])**2,axis=1)/denom[None]
        each.append(ratios)
        for i,h in enumerate(HORIZONS):
            e=a[:,idx,i]-truth[None,idx,i]
            cells.append(dict(year=year,h=h,n=int(idx.sum()),mse=float((e**2).mean()),
                              rmse=float(np.sqrt((e**2).mean())),mae=float(np.abs(e).mean()),
                              relative_MSE=float(ratios[:,i].mean())))
    seed=np.concatenate(each,axis=1).mean(axis=1)
    return float(seed.mean()),seed,cells

def bootstrap(a,b,truth,years,block,draws=2000):
    # Pair all models, horizons and seed-mean loss at the same sampled origins.
    rng=np.random.default_rng(20261015+block)
    e1=((a-truth[None])**2).mean(axis=0);e2=((b-truth[None])**2).mean(axis=0)
    groups=[np.where(years==y)[0] for y in (2020,2021)]
    out=[]
    for _ in range(draws):
        r1=[];r2=[]
        for g in groups:
            n=len(g);starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
            idx=g[np.concatenate([np.arange(s,s+block) for s in starts])[:n]]
            den=(truth[idx]**2).mean(axis=0)
            r1.extend(e1[idx].mean(axis=0)/den);r2.extend(e2[idx].mean(axis=0)/den)
        out.append(100*(1-np.mean(r1)/np.mean(r2)))
    return np.quantile(out,[.025,.975]).tolist()

def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(data_path,out):
    origins,years,truth,pred=panel(data_path,out)
    rows=[];bycell=[];byseed=[];lookup={};seedscores={}
    for name,a in pred.items():
        loss,ss,cells=scores(a,truth,years)
        ens=scores(a.mean(axis=0,keepdims=True),truth,years)[0]
        rows.append(dict(name=name,mean_seed_relative_MSE=loss,prediction_ensemble_relative_MSE=ens,
                         model_seeds=a.shape[0],neural_networks_in_ensemble=(8 if name.startswith('PAIR_') else 4 if a.shape[0]==4 else 0)))
        lookup[name]=rows[-1];seedscores[name]=ss
        bycell.extend([dict(name=name,**r) for r in cells])
        byseed.extend([dict(name=name,seed=(SEEDS[i] if len(ss)==4 else 'deterministic'),score=float(v)) for i,v in enumerate(ss)])
    write_csv(out/'all_scores.csv',rows);write_csv(out/'by_year_horizon.csv',bycell);write_csv(out/'by_seed.csv',byseed)
    selected={m:min((n for n in pred if n.startswith(m+'_l')),key=lambda n:lookup[n]['mean_seed_relative_MSE'])
              for m in ('RESIDUAL','ADDITIVE','FULL','MIXED')}
    cand=selected['MIXED'];cs=lookup[cand]['mean_seed_relative_MSE']
    comparators=[n for n in pred if not n.startswith('MIXED_')]
    same_count=[n for n in comparators if not n.startswith('PAIR_')]
    cstar=min(same_count,key=lambda n:lookup[n]['mean_seed_relative_MSE'])
    cstar_ensemble=min(same_count,key=lambda n:lookup[n]['prediction_ensemble_relative_MSE'])
    gains={n:float(1-cs/lookup[n]['mean_seed_relative_MSE']) for n in (selected['ADDITIVE'],selected['FULL'],'ML_w240',cstar)}
    wins={m:int(np.sum(seedscores[cand]<seedscores[selected[m]])) for m in ('ADDITIVE','FULL')}
    annual={str(y):float(np.mean([r['relative_MSE'] for r in bycell if r['name']==cand and r['year']==y])) for y in (2020,2021)}
    mechanism=all(gains[selected[m]]>=.01 and wins[m]>=3 for m in ('ADDITIVE','FULL'))
    mechanism=bool(mechanism and gains['ML_w240']>=.01 and max(annual.values())<=1.05)
    ensemble_gain=1-lookup[cand]['prediction_ensemble_relative_MSE']/lookup[cstar_ensemble]['prediction_ensemble_relative_MSE']
    accuracy=bool(ensemble_gain>=.01)
    status='DEVELOPMENT_SIGNAL_ONLY_REQUIRES_NOVELTY_AND_HOLDOUT_REVIEW' if mechanism and accuracy else 'STOP_DEVELOPMENT_GATE_FAILED'
    ci={}
    for n in dict.fromkeys(['RW','ML_w240',selected['ADDITIVE'],selected['FULL'],cstar]):
        ci[n]={str(b):bootstrap(pred[cand],pred[n],truth,years,b) for b in (20,40,60)}
    fit_rows=[]
    for path in sorted((out/'fits').glob('*.json')):
        r=json.loads(path.read_text());fit_rows.append({k:v for k,v in r.items() if k not in ('history','history_columns')})
    write_csv(out/'fit_register.csv',fit_rows)
    counts={m:dict(fits=sum(r['mode']==m for r in fit_rows),
                  zero_epoch=sum(r['mode']==m and r['selected_epoch']==0 for r in fit_rows),
                  epochs=sum(r['epochs'] for r in fit_rows if r['mode']==m),
                  summed_fit_wall_seconds=sum(r['seconds'] for r in fit_rows if r['mode']==m)) for m in ('RESIDUAL','ADDITIVE','FULL','MIXED','GRU')}
    with np.load(out/'reference_states.npz',allow_pickle=False) as z:
        rw=float(np.mean(z['w240_selected'][origins]==0))
    result=dict(scientific_status=status,selected=selected,candidate=lookup[cand],
                Cstar_mean_seed=cstar,Cstar_same_count_ensemble=cstar_ensemble,
                gains_mean_seed=gains,paired_seed_wins=wins,annual_relative_MSE=annual,
                ensemble_gain_vs_Cstar=float(ensemble_gain),mechanism_pass=mechanism,accuracy_pass=accuracy,
                reference_RW_fraction=rw,fit_accounting=counts,configuration_count=len(rows),
                origins=451,origins_by_year={str(y):int((years==y).sum()) for y in (2020,2021)},
                primary='Equal year x horizon MSE/RW; mean seed loss is not ensemble loss',
                bootstrap_gain_percent_95=ci,uncertainty_scope='Selected, repeatedly exposed development period; no selection correction; not confirmation',
                price_prefix_end='2021-12-31',later_period_read=False,novelty_established=False)
    dump(out/'RESULTS.json',result)
    dump(out/'STATUS.json',dict(stage='complete_scored',scientific_status=status,later_prices_read=False))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run(args.data,args.out)
