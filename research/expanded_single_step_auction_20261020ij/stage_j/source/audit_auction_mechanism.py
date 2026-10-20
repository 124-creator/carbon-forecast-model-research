#!/usr/bin/env python3
"""Separate arithmetic audit of stage-j saved private predictions and full selection log."""
import argparse,json,math
from pathlib import Path
import numpy as np

MODELS=['rw','price_response','signed_price_only','auction_additive','signed_coverage_interaction_ridge','price_ridge32','auction_ridge32','quadratic_context_ridge','auction_hgb']
PRIMARY='signed_coverage_interaction_ridge'

def metric(pp,obs,rw):
    err=pp-obs;r=rw-obs
    return {'MSE':float(np.mean(err*err)),'RMSE':float(np.linalg.norm(err)/np.sqrt(len(obs))),
            'MAE':float(np.sum(np.abs(err))/len(obs)),'MAPE_percent':float(100*np.sum(np.abs(err)/np.abs(obs))/len(obs)),
            'R2_price':float(1-(err@err)/((obs-obs.mean())@(obs-obs.mean()))),
            'Q_MSE_RW':float((err@err)/(r@r))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--reps',type=int,default=5000);args=p.parse_args()
    root=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    complete=json.loads((root/'COMPLETE.json').read_text());assert complete['status']=='COMPLETE' and complete['source_sha256_unchanged']
    panels=[];maxgap=0
    for rec in complete['results']:
        year=rec['year'];pub=json.loads((root/'cells'/f'{year}.json').read_text());
        with np.load(root/'private'/f'{year}.npz',allow_pickle=False) as z:
            obs=np.asarray(z['true']);rw=np.asarray(z['rw']);pred={key:np.asarray(z[key]) for key in z.files if key not in ('true','rw','origin_record_indices')}
            assert np.isfinite(obs).all() and len(obs)==pub['n_test'] and np.all(np.diff(z['origin_record_indices'])>0)
            for name,v in pred.items():
                old=pub['families'][name.replace('_unshrunk','')]['scores_unshrunk' if name.endswith('_unshrunk') else 'scores_shrunk']
                checks=metric(v,obs,rw)
                for k, vv in checks.items():
                    maxgap=max(maxgap,abs(vv-old[k]));assert abs(vv-old[k])<1e-8,(year,name,k,vv,old[k])
            assert np.allclose(metric(rw,obs,rw)['Q_MSE_RW'],1)
            assert pub['last_train_target_date'] if 'last_train_target_date' in pub else True
            a=pub['last_validation_target_date'];b=pub['first_test_origin_date']
            assert a<b,(year,a,b)
            panels.append((year,obs,rw,pred))
    res={m:float(np.mean([metric(pred[m],y,rw)['Q_MSE_RW'] if m!='rw' else 1 for _,y,rw,pred in panels])) for m in MODELS}
    rng=np.random.default_rng(20261020);stats=[]
    for length in [10,20]:
        boot=np.empty((args.reps,len(MODELS)))
        for j in range(args.reps):
            means=[]
            for yy,y,rw,pred in panels:
                n=len(y);starts=rng.integers(0,n,size=math.ceil(n/length));ids=np.concatenate([(i+np.arange(length))%n for i in starts])[:n]
                den=((rw[ids]-y[ids])**2).mean();assert den>0
                means.append([1.] if False else [((pred[m][ids]-y[ids])**2).mean()/den if m!='rw' else 1 for m in MODELS])
            boot[j,:]=np.mean(means,axis=0)
        base=boot[:,MODELS.index(PRIMARY)]
        for i,m in enumerate(MODELS):
            dif=100*(boot[:,i]-base)
            stats.append({'block':length,'comparator':m,'point_improvement_rw_percentage_points':100*(res[m]-res[PRIMARY]),
                          'marginal95':np.quantile(dif,[.025,.975]).tolist(),
                          'family8_individual99_375':np.quantile(dif,[.003125,.996875]).tolist()})
    final={'status':'PASS_INDEPENDENT_ARITHMETIC_REPLAY','market':'single_EEX_T3PA_auction_market','development_test_years':[y for y,_,_,_ in panels],
           'source_script_frozen':True,'all_six_years':len(panels),'total_test_origins':int(sum(len(y) for _,y,_,_ in panels)),
           'primary':PRIMARY,'mean_Q':res,'max_abs_metric_gap':maxgap,'paired_bootstrap_reps_per_block':args.reps,'intervals':stats,
           'no_new_holdout_claim':True,'market_data_in_public_audit':False}
    (out/'INDEPENDENT_AUDIT.json').write_text(json.dumps(final,indent=2,ensure_ascii=False)+'\n')
    print('AUDIT',final['status'],'n',final['total_test_origins'],'maxgap',maxgap,'primary',res[PRIMARY]);print('comp additive',next(s for s in stats if s['block']==10 and s['comparator']=='auction_additive'))
if __name__=='__main__':main()
