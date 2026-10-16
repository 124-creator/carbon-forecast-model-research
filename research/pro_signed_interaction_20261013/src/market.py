"""Conditional old-development-only market screen. No later-year access."""
from experiment import *

def year_score(arr,true,years):
    loss=np.mean((arr-true[None,:,:])**2,axis=0)
    ratios=[]
    for year in sorted(set(years)):
        m=years==year;ratios.append(loss[m].mean(0)/np.mean(true[m]**2,axis=0))
    return float(np.mean(ratios)),np.asarray(ratios),loss

def bootstrap(arr,comp,true,years,block,B=2000):
    rng=np.random.default_rng(8300+block);loss=np.mean((arr-true[None,:,:])**2,axis=0);closs=np.mean((comp-true[None,:,:])**2,axis=0)
    vals=[];yearset=sorted(set(years))
    for _ in range(B):
        a=[];b=[]
        for yr in yearset:
            idx=np.where(years==yr)[0];n=len(idx);L=min(block,n)
            starts=rng.integers(0,n-L+1,size=int(np.ceil(n/L)))
            samp=idx[np.concatenate([np.arange(s,s+L) for s in starts])[:n]]
            den=np.mean(true[samp]**2,axis=0)
            a.extend((loss[samp].mean(0)/den).tolist());b.extend((closs[samp].mean(0)/den).tolist())
        vals.append(100*(1-np.mean(a)/np.mean(b)))
    return [float(x) for x in np.percentile(vals,[2.5,97.5])]

def run_market(out,synthetic_result):
    gate=json.loads(Path(synthetic_result).read_text())
    if not gate['minimum_gate']:raise RuntimeError('Synthetic minimum gate failed; market execution prohibited')
    out=Path(out)
    if out.exists() and list(out.iterdir()):raise FileExistsError('No overwrite')
    out.mkdir(parents=True,exist_ok=True);(out/'archives').mkdir(exist_ok=True)
    dates,p=load_market();dates=np.asarray(dates);n_initial=int(np.sum(dates<'2020-01-01'))
    if any(d>'2021-12-31' for d in dates):raise ValueError('Future prices crossed model boundary')
    x,loc,raw=prepare(p);hs=training_scales(p,n_initial);bank=filter_bank(p)
    idx=np.asarray([t for t in range(n_initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]],int)
    yy=targets(p,idx);years=np.array([d[:4] for d in dates[idx]])
    refits=np.arange(n_initial,len(p),40);cs=configs();logs=[];results=[];arrays={};begin=time.monotonic()
    for ci,c in enumerate(cs):
        name=cname(c);seeds=nn_seeds(c['family']);ar=np.full((len(seeds),len(idx),3),np.nan);states={}
        for j,seed in enumerate(seeds):
            for k,fit_end in enumerate(refits):
                end=refits[k+1] if k+1<len(refits) else len(p)
                pos=np.where((idx>=fit_end)&(idx<end))[0]
                if not len(pos):continue
                pr,log,state=fit_predict(p,x,loc,raw,hs,c,int(fit_end),idx[pos],seed,window=480,bank=bank)
                ar[j,pos]=pr;log.update(config=name);logs.append(log)
                if state is not None:
                    for key,value in state.items():states[f'{seed}_{fit_end}_{key}']=value
        if not np.isfinite(ar).all():raise FloatingPointError('Missing/nonfinite market predictions')
        score,byyear,_=year_score(ar,yy,years)
        results.append(dict(config=name,family=c['family'],score=score,configuration=c,by_year_relative_mse=byyear.tolist(),years=sorted(set(years)),rmse_h=np.sqrt(np.mean((ar-yy[None,:,:])**2,axis=(0,1))).tolist(),mae_h=np.mean(abs(ar-yy[None,:,:]),axis=(0,1)).tolist(),model_seeds=seeds))
        arrays[name]=ar;np.savez_compressed(out/'archives'/f'{name}.npz',predictions=ar,origins=idx,**states)
        dump(out/'all_results.json',results);dump(out/'fit_logs.json',logs)
        dump(out/'status.json',dict(phase='market_development',completed_configs=ci+1,total_configs=len(cs),last=name,elapsed=time.monotonic()-begin))
        print(f'market {ci+1}/{len(cs)} {name} score={score:.8f}',flush=True)
    best={f:min([r for r in results if r['family']==f],key=lambda r:r['score']) for f in sorted(set(r['family'] for r in results))}
    cand=best['INTERACTION'];cstar=min([r for r in results if r['family']!='INTERACTION'],key=lambda r:r['score'])
    gain=lambda b:1-cand['score']/b['score']
    criteria={'RW':gain(best['RW'])>=.02,'Cstar':gain(cstar)>=.02,'odd_MLP':gain(best['ODD_MLP'])>=.01,'frozen':gain(best['FROZEN'])>=.01,'year_no_large_harm':all(np.mean(row)<=1.05 for row in cand['by_year_relative_mse'])}
    ci={}
    for base in ['RW',cstar['config'],'ODD_MLP','FROZEN']:
        key=best[base]['config'] if base in best else base
        ci[key]={str(L):bootstrap(arrays[cand['config']],arrays[key],yy,years,L) for L in [20,40,60]}
    res=dict(best_by_family=best,candidate=cand,Cstar=cstar,gain_vs_RW=gain(best['RW']),gain_vs_Cstar=gain(cstar),gain_vs_odd_MLP=gain(best['ODD_MLP']),gain_vs_frozen=gain(best['FROZEN']),criteria=criteria,development_gate_passed=all(criteria.values()),configuration_count=len(results),fit_requests=len(logs),initial_train_rows=n_initial,common_origins=len(idx),origins_by_year={yr:int((years==yr).sum()) for yr in sorted(set(years))},max_loaded_date=dates[-1],later_periods_scored=False,hubei_eu_scored=False,bootstrap_gain_percent_95=ci,bootstrap_scope='postselection development only, no multiplicity adjustment, not confirmatory',seconds=time.monotonic()-begin,scientific_status='DEVELOPMENT_PASS_NOT_INDEPENDENT_CONFIRMATION' if all(criteria.values()) else 'STOP_DEVELOPMENT_GATE_FAILED')
    dump(out/'RESULTS.json',res);np.savez_compressed(out/'evaluation_contract.npz',origins=idx,true_changes=yy,dates=dates[idx],horizon_scales=hs)
    print(json.dumps(res,indent=2),flush=True)
    return res

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--synthetic',required=True);a=ap.parse_args()
    with threadpool_limits(limits=1):run_market(a.out,a.synthetic)
