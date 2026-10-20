#!/usr/bin/env python3
"""Independent standard-metric and paired block statistical audit from private heldout arrays.

Public export excludes date-level market rows and per-origin predictions.
"""
from __future__ import annotations
import argparse, json, hashlib, math
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score

CANDIDATE='comp_zero'
COMPARISONS=['rw','static','ridge32','hgb','raw_free','raw_zero','comp_free','gru_residual']

def metrics(pred,obs,rw_mse):
    err=pred-obs
    mse=np.mean(err**2)
    return {'RMSE':float(np.sqrt(mse)),'MAE':float(np.mean(np.abs(err))),
            'MAPE_percent':float(np.mean(100*np.abs(err)/np.maximum(np.abs(obs),1e-8))),
            'R2_price_level':float(r2_score(obs,pred)),'Q_MSE_over_RW':float(mse/rw_mse),'MSE':float(mse)}

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--run',required=True);pa.add_argument('--out',required=True);pa.add_argument('--blocks',nargs='+',type=int,default=[10,20]);pa.add_argument('--reps',type=int,default=5000);a=pa.parse_args()
    src=Path(a.run);dst=Path(a.out);dst.mkdir(parents=True,exist_ok=False)
    comp=json.loads((src/'COMPLETE.json').read_text())
    if comp['status']!='COMPLETE_DECLARED_MATRIX':raise ValueError('Not complete')
    panels=[];max_gap=0.0;metrics_public=[];exposed=0
    for r in comp['results']:
        if r['status']!='COMPLETE':continue
        f=src/'private'/(r['cell']+'.npz')
        with np.load(f,allow_pickle=False) as z:
            truth=z['observed'];rw=z['anchor'];mse_rw=np.mean((truth-rw)**2)
            if mse_rw<1e-12:raise ValueError('Invalid RW denominator')
            allpred={key:z[key] for key in z.files if key not in ('observed','origins','anchor')}
            observed={key:metrics(pred,truth,mse_rw) for key,pred in allpred.items() if not key.startswith('seeds_')}
            for k,pred in observed.items():
                for stat,val in pred.items():
                    old=r['score'][k][stat]
                    max_gap=max(max_gap,abs(val-old))
                    if abs(val-old)>1e-9: raise ValueError(f'Score mismatch {r["cell"]}/{k}/{stat} {val}/{old}')
            for k,pred in allpred.items():
                if not k.startswith('seeds_'):continue
                m=k[6:]; seeds=np.asarray(pred)
                if seeds.shape[0]!=3:raise ValueError('Expected exactly three seed forecasts')
                sm=[metrics(p,truth,mse_rw) for p in seeds]
                want=np.mean([mm['Q_MSE_over_RW'] for mm in sm]);got=r['score'][m]['Q_seed_mean']
                if abs(got-want)>1e-9:raise ValueError('Seed mean mismatch '+r['cell']+'/'+m)
                if not np.allclose(seeds.mean(0),allpred[m],rtol=0,atol=1e-9):raise ValueError('Ensemble mismatch')
                metrics_public.append({'cell':r['cell'],'model':m,'main_seed_mean':want, 'ensemble_Q':observed[m]['Q_MSE_over_RW']})
            panels.append({'cell':r['cell'],'obs':truth,'pred':allpred,'rw':rw,'n':len(truth), 'source_group':r['info']['market']})
            exposed+=len(truth)
    print('Replayed',len(panels),'panels',exposed,'origins', 'max absolute numeric gap',max_gap,flush=True)
    modelset=[k for k in panels[0]['pred'] if not k.startswith('seeds_')]
    summary={}
    for model in modelset:
        means=[]
        for p in panels:
            mse_rw=np.mean((p['obs']-p['rw'])**2)
            if 'seeds_'+model in p['pred']:
                ms=[metrics(x,p['obs'],mse_rw) for x in p['pred']['seeds_'+model]]
                q=np.mean([m['Q_MSE_over_RW'] for m in ms]);mae=np.mean([m['MAE'] for m in ms]);r2=np.mean([m['R2_price_level'] for m in ms])
            else:
                met=metrics(p['pred'][model],p['obs'],mse_rw);q=met['Q_MSE_over_RW'];mae=met['MAE'];r2=met['R2_price_level']
            means.append({'cell':p['cell'],'q':float(q),'mae':float(mae),'r2':float(r2)})
        summary[model]={'mean_Q_equal_cell':float(np.mean([d['q'] for d in means])),'cell_scores':means}
    # At a given block length, preserve same circular block positions for all models and horizons.
    # Mean-seed loss is primary. Recomputed RW denominator each bootstrap draw.
    rng=np.random.default_rng(20261020)
    boot={}
    for block in a.blocks:
        imp={model:np.zeros(a.reps) for model in COMPARISONS}
        for i in range(a.reps):
            cell_effect={model:[] for model in COMPARISONS}
            for p in panels:
                n=p['n'];nstarts=int(math.ceil(n/block))
                starts=rng.integers(0,n,size=nstarts)
                ids=np.concatenate([(start+np.arange(block))%n for start in starts])[:n]
                den=float(np.mean((p['obs'][ids]-p['rw'][ids])**2))
                if den<=1e-12:raise ValueError('zero RW bootstrap denom')
                def seeded_loss(m):
                    if 'seeds_'+m in p['pred']:
                        predictions=p['pred']['seeds_'+m]
                        return float(np.mean([np.mean((pp[ids]-p['obs'][ids])**2) for pp in predictions]))
                    return float(np.mean((p['pred'][m][ids]-p['obs'][ids])**2))
                cand=seeded_loss(CANDIDATE)/den
                for m in COMPARISONS:cell_effect[m].append(100*(seeded_loss(m)/den-cand))
            for m in COMPARISONS:imp[m][i]=np.mean(cell_effect[m])
        for m in COMPARISONS:
            name=m+f'_block{block}'
            boot[name]={'n_replications':a.reps,'block':block,
                        'point_effect_rw_percentage_points':100*(summary[m]['mean_Q_equal_cell']-summary[CANDIDATE]['mean_Q_equal_cell']),
                        'marginal_95':np.quantile(imp[m],[.025,.975]).tolist(),
                        'eight_test_bonferroni_99_375':np.quantile(imp[m],[.003125,.996875]).tolist()}
    result={'status':'PASS_REPLAY_AND_STATISTICS','n_panels':len(panels),'total_evaluated_origins':exposed,'max_aggregate_score_gap':max_gap,
            'all_model_scores':summary,'bootstrap':boot,'primary':CANDIDATE,'comparison_basis':'mean_seed_loss, not ensemble, for all neural models; 1step price MSE/RW-MSE equal across cells',
            'warning':'development/exploratory; common dates and same instrument across two EU sources cause dependence; bootstrap conditions on cells and does not cover all model-search history',
            'raw_and_per_origin_scores_not_public':True}
    (dst/'INDEPENDENT_REPLAY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Mean Q ranking:',sorted([(k,round(v['mean_Q_equal_cell'],6)) for k,v in summary.items()],key=lambda x:x[1]))
    print('candidate vs static',boot['static_block10'])
if __name__=='__main__':main()
