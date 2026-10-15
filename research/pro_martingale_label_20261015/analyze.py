"""Fixed factorial metrics; seed-loss means never conflated with ensembles."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import core
from experiment import jwrite

HS=[0,4,19]
def cell_scores(pred,truth,year):
    loss=(pred[:,:,HS]-truth[None,:,HS])**2
    scores=np.empty((pred.shape[0],6));cells=[];n=0
    for y in ('2020','2021'):
        mask=year==y
        for k,h in enumerate(HS):
            mse=loss[:,mask,k].mean(1);den=(truth[mask,h]**2).mean();scores[:,n]=mse/den
            cells.append(dict(year=y,horizon=h+1,n=int(mask.sum()),ratio=float(scores[:,n].mean()),RMSE_seed_loss=float(np.sqrt(mse.mean())),MAE_seed_mean=float(np.abs(pred[:,mask,h]-truth[None,mask,h]).mean())))
            n+=1
    return scores,cells

def block_indices(rng,n,length):
    count=int(np.ceil(n/length));starts=rng.integers(0,n-length+1,size=count)
    return (starts[:,None]+np.arange(length)).ravel()[:n]

def run(out):
    out=Path(out)
    with np.load(out/'EVALUATION.npz',allow_pickle=False) as z:
        mask=z['eligible'];truth=z['truth'][mask];year=z['years'][mask]
    preds={};results={}
    for path in sorted((out/'predictions').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:pred=z['prediction'][:,mask].copy()
        if not np.isfinite(pred[:,:,HS]).all():raise ValueError(f'Unfitted evaluated horizon {path}')
        name=path.stem;preds[name]=pred;s,c=cell_scores(pred,truth,year);e,_=cell_scores(pred.mean(0,keepdims=True),truth,year)
        results[name]=dict(score=float(s.mean()),ensemble=float(e.mean()),seed_scores=s.mean(1).tolist(),cells=c)
    controls=['TWOHOT','GAUSS','CAT_MSE','IND_DIFF','FIXED6','RANK6','WIDE_RANK6']
    candidate=results['MP_DIFF'];pool=[k for k in results if k!='MP_DIFF']
    best=min(pool,key=lambda k:results[k]['score']);best_e=min(pool,key=lambda k:results[k]['ensemble'])
    gains={k:1-candidate['score']/results[k]['score'] for k in controls+['RW',best]}
    wins={k:int(np.sum(np.array(candidate['seed_scores'])<np.array(results[k]['seed_scores']))) for k in controls}
    annual={y:float(np.mean([c['ratio'] for c in candidate['cells'] if c['year']==y])) for y in ('2020','2021')}
    mechanism=(all(gains[k]>=.01 for k in ['TWOHOT','GAUSS','CAT_MSE']) and all(gains[k]>0 for k in ['IND_DIFF','FIXED6','RANK6','WIDE_RANK6']) and wins['TWOHOT']>=3 and wins['GAUSS']>=3)
    accuracy=(gains['RW']>=.02 and gains[best]>0 and candidate['ensemble']<results[best_e]['ensemble'] and max(annual.values())<=1.05)
    summary=dict(status='DEVELOPMENT_LEAD_ONLY' if mechanism and accuracy else 'STOP_DEVELOPMENT_GATE_FAILED',candidate=candidate,best_non_candidate=best,best_ensemble_non_candidate=best_e,gains=gains,seed_wins=wins,annual=annual,mechanism_pass=bool(mechanism),accuracy_pass=bool(accuracy),new_unseen_test=False,novelty_established=False)
    with (out/'all_scores.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','mean_seed_relative_MSE','ensemble_relative_MSE','number_of_seed_models'])
        for k,r in results.items():w.writerow([k,r['score'],r['ensemble'],len(r['seed_scores'])])
    jwrite(out/'ALL_MODEL_METRICS.json',results)
    # One paired-block draw applied to all models and horizons within each year.
    targets=list(dict.fromkeys(controls+[best,'RW']))
    l={k:((preds[k][:,:,HS]-truth[None,:,HS])**2).mean(0) for k in ['MP_DIFF']+targets}
    y2=truth[:,HS]**2;intervals={k:{} for k in targets}
    for length in (20,40,60):
        rng=np.random.default_rng(314159+length);vals={k:[] for k in targets}
        groups=[np.flatnonzero(year==y) for y in ('2020','2021')]
        for _ in range(2000):
            ix=[g[block_indices(rng,len(g),length)] for g in groups]
            stat={k:float(np.mean(np.concatenate([l[k][i].mean(0)/y2[i].mean(0) for i in ix]))) for k in l}
            for k in targets:vals[k].append(100*(1-stat['MP_DIFF']/stat[k]))
        for k in targets:intervals[k][str(length)]=np.quantile(vals[k],[.025,.975]).tolist()
    summary['descriptive_block_intervals_percent']=intervals
    summary['interval_scope']='Repeatedly exposed development; no cross-stage selection correction or confirmatory coverage claim'
    jwrite(out/'RESULTS.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);o=a.parse_args();run(o.out)
