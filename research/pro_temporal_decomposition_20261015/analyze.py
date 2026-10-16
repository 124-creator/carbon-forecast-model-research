"""Prespecified development scoring and paired block uncertainty, no model fitting."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from model import MODES,SEEDS
from run_market import dump
H=[1,5,20]

def score_losses(loss,truth,years):
    return float(np.mean([np.mean(loss[years==yr,j])/np.mean(truth[years==yr,j]**2) for yr in ('2020','2021') for j in range(3)]))

def block_indices(n,length,rng):
    starts=rng.integers(0,max(n-length+1,1),size=int(np.ceil(n/length)))
    return np.concatenate([np.arange(s,min(s+length,n)) for s in starts])[:n]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();r=Path(a.out)
    with np.load(r/'TARGETS.npz',allow_pickle=False) as z:
        truth=z['truth'][:,[0,4,19]];years=z['years'];orig=z['origins']
    predictions={}
    with np.load(r/'NEURAL_PREDICTIONS.npz',allow_pickle=False) as z:
        for mode in MODES:predictions[mode]=np.stack([z[f'{mode}_s{s}'][:,[0,4,19]] for s in SEEDS])
    with np.load(r/'BASELINE_PREDICTIONS.npz',allow_pickle=False) as z:
        predictions.update({k:z[k][None] for k in z.files})
    table=[];losses={};ensloss={};details=[];seed_scores={}
    for name,pred in predictions.items():
        errors=pred-truth[None];lo=np.mean(errors**2,0);en=(pred.mean(0)-truth)**2
        losses[name]=lo;ensloss[name]=en
        seed_scores[name]=[score_losses(e**2,truth,years) for e in errors]
        score=score_losses(lo,truth,years);ensemble=score_losses(en,truth,years)
        yrscore={yr:float(np.mean([np.mean(lo[years==yr,j])/np.mean(truth[years==yr,j]**2) for j in range(3)])) for yr in ('2020','2021')}
        table.append(dict(model=name,primary_seed_mean_MSE_ratio=score,seed_ensemble_MSE_ratio=ensemble,seed_count=len(pred),**{f'year_{y}':v for y,v in yrscore.items()}))
        for label,err in [('seed_mean',errors),('prediction_ensemble',errors.mean(0)[None])]:
            for yr in ('2020','2021'):
                for j,h in enumerate(H):
                    cell=err[:,years==yr,j];mse=float(np.mean(cell**2))
                    details.append(dict(model=name,statistic=label,year=yr,h=h,n_origins=int(sum(years==yr)),MSE=mse,RMSE=float(np.sqrt(mse)),MAE=float(np.mean(abs(cell))),relative_MSE=mse/float(np.mean(truth[years==yr,j]**2))))
    for fn,rows in [('all_scores.csv',table),('by_year_horizon.csv',details)]:
        with (r/fn).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lookup={row['model']:row for row in table};candidate='D_TEMP';rr={k:v['primary_seed_mean_MSE_ratio'] for k,v in lookup.items()}
    Cstar=min((k for k in rr if k!=candidate),key=rr.get)
    gains={k:1-rr[candidate]/rr[k] for k in ['RW','D_ERM','S_TEMP','S_ERM','D_JOINT','GRU',Cstar]}
    wins={k:int(sum(x<y for x,y in zip(seed_scores[candidate],seed_scores[k]))) for k in ['D_ERM','S_TEMP']}
    interaction=(rr['D_ERM']-rr['D_TEMP'])-(rr['S_ERM']-rr['S_TEMP'])
    gates=dict(vs_RW=gains['RW']>=.02,vs_D_ERM=gains['D_ERM']>=.01,vs_S_TEMP=gains['S_TEMP']>=.01,
        not_worse_S_ERM=rr[candidate]<=rr['S_ERM'],positive_interaction=interaction>0,seed_wins=all(v>=3 for v in wins.values()))
    mechanism=all(gates.values());cstar_gain=1-rr[candidate]/rr[Cstar]
    ensemble_gain=1-lookup[candidate]['seed_ensemble_MSE_ratio']/lookup[Cstar]['seed_ensemble_MSE_ratio']
    predgate=mechanism and cstar_gain>=.01 and ensemble_gain>=.01 and max(lookup[candidate]['year_2020'],lookup[candidate]['year_2021'])<=1.05
    # All resamples remain developmental, not corrected for model selection.
    compare=list(dict.fromkeys(['RW','D_ERM','S_TEMP','S_ERM','D_JOINT','GRU',Cstar]));ci={}
    for L in (20,40,60):
        rng=np.random.default_rng(719+L);samples={k:[] for k in compare};ii=[]
        for repeat in range(2000):
            sc={k:[] for k in set(compare+['D_TEMP'])}
            for yr in ('2020','2021'):
                at=np.flatnonzero(years==yr);sel=at[block_indices(len(at),L,rng)]
                rw=np.mean(truth[sel]**2,0)
                for k in sc:sc[k].extend((np.mean(losses[k][sel],0)/rw).tolist())
            sc={k:float(np.mean(v)) for k,v in sc.items()}
            for k in compare:samples[k].append(100*(1-sc['D_TEMP']/sc[k]))
            ii.append((sc['D_ERM']-sc['D_TEMP'])-(sc['S_ERM']-sc['S_TEMP']))
        ci[str(L)]={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()}
        ci[str(L)]['factorial_interaction']=np.quantile(ii,[.025,.975]).tolist()
    fits=[json.loads(p.read_text()) for p in sorted((r/'fits').glob('*.json'))];training={}
    for mode in MODES:
        f=[x for x in fits if x['mode']==mode]
        training[mode]=dict(fits=len(f),sum_CPU_seconds=sum(x['cpu_seconds'] for x in f),sum_fit_wall_seconds=sum(x['wall_seconds'] for x in f),sum_warmup_steps=sum(len(x['warmup_history']) for x in f),sum_router_steps=sum(len(x['router_history']) for x in f),sum_final_epochs=sum(x['final_epoch_count'] for x in f),selected_zero=sum(x['selected_epoch']==0 for x in f),forward_sample_visits=sum(x['forward_sample_visits'] for x in f),parameters=f[0]['parameters'])
    result=dict(status='MECHANISM_PASSED_REQUIRE_FRESH_STRONG_ENSEMBLE' if mechanism else 'STOP_DEVELOPMENT_MECHANISM_FAILED',mechanism_gate=mechanism,prediction_gate_pre_ensemble=predgate,gates=gates,candidate=candidate,Cstar=Cstar,primary=rr,ensemble={k:v['seed_ensemble_MSE_ratio'] for k,v in lookup.items()},gains_vs_comparators=gains,paired_seed_wins=wins,seed_scores=seed_scores,factorial_interaction=interaction,paired_block_CI_percent=ci,training=training,neural_fits=len(fits),baseline_configurations=23,origins=len(orig),origin_counts={y:int(sum(years==y)) for y in ('2020','2021')},later_history_access=False,secondary_market_fits=False,novelty_established=False,scope='fixed development pilot; old exposed 2020/2021 segment; no selection/multiplicity correction; baselines regenerated',strong_online_ensemble_executed=False)
    dump(r/'RESULTS.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
