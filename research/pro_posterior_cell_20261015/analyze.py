"""Prespecified scoring, paired dependent bootstrap and stop decision."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import core
from experiment import write_json

def score(pred,truth,years,mask):
    pred=pred[:,mask];y=truth[mask];yr=years[mask];cells=[];perseed=[]
    for year in ('2020','2021'):
        ii=yr==year
        for h in core.HORIZONS:
            err=pred[:,ii,h-1]-y[ii,h-1][None];den=np.mean(y[ii,h-1]**2)
            mm=np.mean(err**2,axis=1)/den
            cells.append({'year':year,'h':h,'n':int(ii.sum()),'relative_MSE':float(mm.mean()),
                          'RMSE_seed_mean':float(np.sqrt(np.mean(err**2,axis=1)).mean()),
                          'MAE_seed_mean':float(abs(err).mean())})
            perseed.append(mm)
    ps=np.mean(perseed,axis=0)
    ec=[]
    for year in ('2020','2021'):
        ii=yr==year
        for h in core.HORIZONS:
            ec.append(np.mean((pred.mean(0)[ii,h-1]-y[ii,h-1])**2)/np.mean(y[ii,h-1]**2))
    return {'score':float(ps.mean()),'ensemble':float(np.mean(ec)),'seeds':ps.tolist(),'cells':cells}

def bootstrap(preds,truth,years,mask,base_name,block,draws=2000):
    cand=preds['INTEGRATED'][:,mask];base=preds[base_name][:,mask];y=truth[mask];yr=years[mask]
    hs=np.array(core.HORIZONS)-1
    errors=[(a[:,:,hs]-y[None,:,hs])**2 for a in (cand,base)]
    es=[a.mean(0) for a in errors];sq=y[:,hs]**2
    rng=np.random.default_rng(68000+block);gain=[]
    for _ in range(draws):
        cs=[];bs=[]
        for year in ('2020','2021'):
            idx=np.flatnonzero(yr==year);n=len(idx);length=min(n,block)
            starts=rng.integers(0,n-length+1,size=int(np.ceil(n/length)))
            take=idx[np.concatenate([np.arange(t,t+length) for t in starts])[:n]]
            den=np.mean(sq[take],axis=0)
            cs.extend(np.mean(es[0][take],0)/den);bs.extend(np.mean(es[1][take],0)/den)
        gain.append(100*(1-np.mean(cs)/np.mean(bs)))
    return [float(a) for a in np.quantile(gain,[.025,.975])]

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--out',required=True);args=pa.parse_args();out=Path(args.out)
    with np.load(out/'evaluation.npz',allow_pickle=False) as z:truth=z['truth'];years=z['years'];mask=z['score_mask']
    with np.load(out/'predictions.npz',allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
    scores={k:score(v,truth,years,mask) for k,v in pred.items()}
    cs=min((k for k in scores if k!='INTEGRATED'),key=lambda k:scores[k]['score'])
    ce=min((k for k in scores if k!='INTEGRATED'),key=lambda k:scores[k]['ensemble'])
    c=scores['INTEGRATED'];gains={k:1-c['score']/v['score'] for k,v in scores.items() if k!='INTEGRATED'}
    wins={k:int(np.sum(np.array(c['seeds'])<np.array(scores[k]['seeds']))) for k in ('POINT','FREE_DAMP','CONSTANT','FULL','GRU')}
    annual={y:float(np.mean([v['relative_MSE'] for v in c['cells'] if v['year']==y])) for y in ('2020','2021')}
    mech=gains['POINT']>=.01 and gains['FREE_DAMP']>=.01 and wins['POINT']>=3 and wins['FREE_DAMP']>=3 and gains['CONSTANT']>0
    acc=gains['RW']>=.02 and gains[cs]>=.02 and c['ensemble']<scores[ce]['ensemble'] and max(annual.values())<=1.05
    result={'status':'DEVELOPMENT_PASS_PENDING_INDEPENDENT_CONFIRMATION' if mech and acc else 'STOP_DEVELOPMENT_GATE_FAILED',
            'candidate':c,'Cstar_mean':cs,'Cstar_ensemble':ce,'gains':gains,'paired_seed_wins':wins,'annual':annual,
            'mechanism_pass':bool(mech),'accuracy_pass':bool(acc),'novelty_established':False,
            'price_prefix_end':'2021-12-31','later_history_read':False,'score_count':len(scores),
            'uncertainty_scope':'Repeatedly exposed development, no selection correction or independent confirmation'}
    targets=list(dict.fromkeys(['RW','POINT','FREE_DAMP','CONSTANT',cs]))
    result['gain_percent_CI95']={k:{str(b):bootstrap(pred,truth,years,mask,k,b) for b in (20,40,60)} for k in targets}
    write_json(out/'ALL_SCORES.json',scores);write_json(out/'RESULTS.json',result)
    with (out/'all_scores.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','mean_seed_relative_MSE','prediction_ensemble_relative_MSE','seed_count'])
        for k,v in scores.items():w.writerow([k,v['score'],v['ensemble'],len(v['seeds'])])
    write_json(out/'STATUS.json',{'stage':'finished','scientific_status':result['status']})
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
