"""One-shot evaluation after family choices frozen using 2019/2020 only."""
from run_market import *
from select_and_ablate import select
from datetime import datetime,timezone

def main():
    if (O/'evaluation_complete.json').exists():
        raise RuntimeError('Evaluation already saved; use a fresh output directory for reproduction')
    sel=select();d=load();registry=configs();ref=json.loads((O/'candidate_frozen_before_check.json').read_text())
    if ref['id']!=sel['conditional']['id']:raise RuntimeError('Candidate selection changed')
    frozen={x['id']:hashlib.sha256((P/(x['id']+'.npz')).read_bytes()).hexdigest() for x in registry}
    (O/'check_open_receipt.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'choices_before_check':sel,'archive_sha256':frozen},indent=2))
    chosen={fam:np.load(P/(c['id']+'.npz'))['delta'] for fam,c in sel.items() if not fam.startswith('_')}
    mainparams=sel['conditional']['params']
    for fam in ['free','global']:
        c=next(c for c in registry if c['family']==fam and c['params']==mainparams)
        chosen[fam+'_matched']=np.load(P/(c['id']+'.npz'))['delta']
    for fam in ['no_gap','fixed','no_drift']:
        chosen['ablation_'+fam]=np.load(P/('ablation_'+fam+'.npz'))['delta']
    records=[];overall=[];losses={};forecast_rows=[];seedrows=[]
    for name,pred in chosen.items():
        stats=score(d,pred,[2021,2022]);records.extend([dict(model=name,**r) for r in stats]);errbyyear={}
        for year in [2021,2022]:
            ix=origin_mask(d,year);errbyyear[year]=np.mean((pred[:,ix]-d.y[ix][None])**2,axis=0)
            for ss in range(len(pred)):
                sr=score(d,pred[ss:ss+1],[year]);seedrows.append({'model':name,'year':year,'seed':([11,29,47][ss] if len(pred)>1 else 'deterministic'),'relative_mse':np.mean([x['relative_mse'] for x in sr])})
                for i in ix:
                    for j,h in enumerate(H):forecast_rows.append({'model':name,'seed':([11,29,47][ss] if len(pred)>1 else 'deterministic'),'origin_index':int(i),'origin_date':str(d.dates[i])[:10],'horizon':int(h),'target_index':int(i+h),'target_date':str(d.dates[i+h])[:10],'price_at_origin':d.p[i],'target_price':d.p[i+h],'forecast_price':d.p[i]+pred[ss,i,j],'squared_error':float((pred[ss,i,j]-d.y[i,j])**2)})
        losses[name]=errbyyear
        rm=float(np.mean([r['relative_mse'] for r in stats]));overall.append({'model':name,'relative_mse':rm,'skill_vs_RW_pct':100*(1-rm),'NMSE':np.mean([r['nmse'] for r in stats]),'ratio_2021':np.mean([r['relative_mse'] for r in stats if r['year']==2021]),'ratio_2022':np.mean([r['relative_mse'] for r in stats if r['year']==2022])})
    pd.DataFrame(records).to_csv(O/'check_per_year_horizon.csv',index=False)
    pd.DataFrame(overall).to_csv(O/'check_summary.csv',index=False)
    pd.DataFrame(seedrows).to_csv(O/'check_per_seed.csv',index=False)
    pd.DataFrame(forecast_rows).to_csv(O/'selected_forecasts_check.csv.gz',index=False,compression='gzip')
    # Selected comparison from TUNING only; no post-test switch of comparator.
    cstar=min((k for k in sel if k not in ['conditional','_receipt']),key=lambda k:sel[k]['tuning_relative_mse'])
    boot=[];rng=np.random.default_rng(20261013)
    comparisons=list(dict.fromkeys(['RW',cstar,'free','free_matched','global','global_matched']))
    for length in [20,40,60]:
        estimates={m:[] for m in comparisons}
        for b in range(2000):
            totals={m:[] for m in comparisons+['conditional']}
            for year in [2021,2022]:
                n=len(losses['RW'][year]);starts=rng.integers(0,n-length+1,int(np.ceil(n/length)));idx=np.concatenate([np.arange(s,s+length) for s in starts])[:n]
                base=losses['RW'][year][idx].mean(axis=0)
                for m in totals:totals[m].append(float(np.mean(losses[m][year][idx].mean(axis=0)/base)))
            cand=np.mean(totals['conditional'])
            for m in comparisons:estimates[m].append(100*(1-cand/np.mean(totals[m])))
        for m,v in estimates.items():
            lo,hi=np.percentile(v,[2.5,97.5]);boot.append({'comparator':m,'block_length':length,'draws':2000,'skill_lo_pct':lo,'skill_hi_pct':hi,'purpose':'exploratory frozen-snapshot historical comparison; no multiple-comparison or selection correction'})
    pd.DataFrame(boot).to_csv(O/'paired_block_intervals.csv',index=False)
    sy=pd.read_csv(O/'synthetic_per_path.csv');summary=sy.groupby(['process','model']).relative_mse.agg(['mean','std','count']).reset_index();summary.to_csv(O/'synthetic_summary.csv',index=False)
    # Parameter diagnostics, never used for hyperparameter choice.
    taus=np.load(P/(sel['conditional']['id']+'.npz'))['tau'];tr=[]
    for year in [2021,2022]:
        ids=origin_mask(d,year)
        for j,seed in enumerate([11,29,47]):
            tt=taus[j,ids];tr.append(dict(year=year,seed=seed,n=len(tt),minimum=float(tt.min()),median=float(np.median(tt)),maximum=float(tt.max()),fraction_below1=float(np.mean(tt<1)),fraction_above39=float(np.mean(tt>39))))
    pd.DataFrame(tr).to_csv(O/'tau_diagnostics.csv',index=False)
    tab=pd.DataFrame(overall).set_index('model');cand=float(tab.loc['conditional','relative_mse']);basefamilies=[k for k in sel if not k.startswith('_') and k!='conditional'];best=min(basefamilies,key=lambda k:tab.loc[k,'relative_mse']);gain=100*(1-cand/tab.loc[best,'relative_mse']);g_global=100*(1-cand/tab.loc['global','relative_mse']);yearok=max(tab.loc['conditional','ratio_2021'],tab.loc['conditional','ratio_2022'])<=1.05
    decision={'candidate':sel['conditional'],'Cstar_tuning_selected':cstar,'strongest_non_candidate_descriptive_check':best,'skill_vs_RW_pct':100*(1-cand),'skill_vs_strongest_check_pct':gain,'skill_vs_global_pct':g_global,'year_safety_gate':bool(yearok),'advance':bool(100*(1-cand)>=2 and gain>=2 and g_global>=1 and yearok),'all_configurations':len(registry),'primary_neural_trajectories':36,'diagnostic_neural_trajectories':9,'synthetic_neural_trajectories':45,'test_years':[2021,2022],'old_EU_holdout_scored':False,'data_authentication_sufficient_for_official_market_claim':False,'utc_finished':datetime.now(timezone.utc).isoformat()}
    (O/'decision.json').write_text(json.dumps(decision,indent=2));(O/'evaluation_complete.json').write_text(json.dumps({'completed':True,'selected_families':len(basefamilies)+1,'configured_archives':len(registry),'no_model_changes_after_check':True},indent=2))
    print(pd.DataFrame(overall).sort_values('relative_mse').to_string(index=False));print(json.dumps(decision,indent=2));print(summary.to_string(index=False))
if __name__=='__main__':main()
