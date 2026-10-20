#!/usr/bin/env python3
"""Independent arithmetic and paired bootstrap audit; no training imports.
Test the complete saved origin arrays, leave historical outcomes unchanged.
"""
import argparse, json, numpy as np, math
from pathlib import Path

MODELS=['rw','reference_direct','blend_constant','auction_price_only_ridge','additive_spread_ridge',
        'spread_context_interaction_ridge','fixed_spread_gate','bounded_depth_conditioned_spread_gate','hgb_same_information']
MAIN='bounded_depth_conditioned_spread_gate'

def calc(pred,y,rw):
    e=pred-y
    den=np.mean((rw-y)**2)
    return {'MSE':float(e@e/len(e)),'RMSE':float(np.linalg.norm(e)/np.sqrt(len(e))),
            'MAE':float(np.sum(np.abs(e))/len(e)),'MAPE_percent':float(100*np.sum(np.abs(e)/np.abs(y))/len(e)),
            'R2_price':float(1-(e@e)/((y-y.mean())@(y-y.mean()))),
            'Q_MSE_RW':float((e@e)/len(e)/den)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--reps',type=int,default=5000);a=p.parse_args()
    root=Path(a.run); out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    old=json.loads((root/'COMPLETE.json').read_text());assert old['status']=='COMPLETE' and old['cells']==12 and old['source_code_unchanged']
    panels={k:[] for k in ('asof_close','strict_prior')};maxgap=0
    for panel in old['outcomes']:
        mode=panel['mode'];year=panel['year'];file=root/'private'/f'{mode}_{year}.npz'
        with np.load(file,allow_pickle=False) as z:
            y=z['observed'];rw=z['rw'];assert len(y)==panel['n'] and np.isfinite(y).all() and np.all(np.diff(z['origins'])>0)
            predictions={m:z[m] if m!='rw' else rw for m in MODELS}
            for model,pred in predictions.items():
                sc=calc(pred,y,rw)
                for stat,value in sc.items():
                    saved=panel['scores'][model][stat];maxgap=max(maxgap,abs(value-saved))
                    if abs(value-saved)>1e-9:raise RuntimeError('Score mismatch '+mode+str(year)+model+stat)
            for extra in z.files:
                if not extra.endswith('_unshrunk'):continue
                sc=calc(z[extra],y,rw)
                for stat,value in sc.items():
                    saved=panel['scores'][extra][stat];maxgap=max(maxgap,abs(value-saved))
                    if abs(value-saved)>1e-9:raise RuntimeError('unshrunk score mismatch')
            receipt=json.loads((root/'cells'/f'{mode}_{year}.json').read_text());assert receipt['last_validation_label_date']<receipt['first_test_origin_date']
            if receipt['source_quote_available_share_test']<1:raise RuntimeError('Missing coverage on 2018-23 test range')
            panels[mode].append((year,y,rw,predictions))
    summary={};effects=[]
    for mode,cells in panels.items():
        means={k:float(np.mean([calc(P[k],y,rw)['Q_MSE_RW'] for yr,y,rw,P in cells])) for k in MODELS}
        info={'count_cells':len(cells),'origins':sum(len(y) for _,y,_,_ in cells),'mean_Q':means,'mean_price_R2':{k:float(np.mean([calc(P[k],y,rw)['R2_price'] for yr,y,rw,P in cells])) for k in MODELS},
              'mean_MAPE_percent':{k:float(np.mean([calc(P[k],y,rw)['MAPE_percent'] for yr,y,rw,P in cells])) for k in MODELS}}
        summary[mode]=info
        rng=np.random.default_rng(20261020 + (0 if mode=='asof_close' else 100))
        for block in (10,20):
            boot=np.empty((a.reps,len(MODELS)),dtype=float)
            for j in range(a.reps):
                scores=[]
                for year,y,rw,pr in cells:
                    n=len(y);starts=rng.integers(0,n,size=math.ceil(n/block));idx=np.concatenate([(start+np.arange(block))%n for start in starts])[:n]
                    denom=np.mean((rw[idx]-y[idx])**2)
                    if denom<=1e-12:raise RuntimeError('Invalid denominator')
                    scores.append([np.mean((pr[k][idx]-y[idx])**2)/denom for k in MODELS])
                boot[j]=np.mean(scores,axis=0)
            primary=boot[:,MODELS.index(MAIN)]
            for i,k in enumerate(MODELS):
                difference=100*(boot[:,i]-primary)
                effects.append({'mode':mode,'block':block,'comparator':k,
                                'model_gain_over_comparator_rw_percentage_points':100*(means[k]-means[MAIN]),
                                'marginal95':np.quantile(difference,[.025,.975]).tolist(),
                                'family8_individual99_375':np.quantile(difference,[.003125,.996875]).tolist()})
    result={'status':'PASS_REPLAY_AND_BOOTSTRAP','modes':summary,'max_absolute_metric_gap':maxgap,'intervals':effects,
            'bootstrap_reps_per_mode_and_block':a.reps,
            'source_information_condition':'asof_close price unverified exact first publication time; strict_prior is earlier than current auction date',
            'price_source_identity':'author-reported EUA_LONG, market venue / quote definition not yet independently authenticated',
            'all_market_years_exposed_development':True,'public_release_excludes_raw_price_and_per_origin_predictions':True,
            'not_evidence_of_independent_method_novelty':True}
    (out/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('AUDIT',result['status'],'maxgap',maxgap)
    for mode in panels:
        print('MAIN',mode,summary[mode]['mean_Q'][MAIN],
              'direct',summary[mode]['mean_Q']['reference_direct'],
              'fixed',summary[mode]['mean_Q']['fixed_spread_gate'])
        for k in ('rw','reference_direct','fixed_spread_gate'):
            print('INTERVAL',mode,k,next(x for x in effects if x['mode']==mode and x['block']==10 and x['comparator']==k)['marginal95'])
if __name__=='__main__':main()
