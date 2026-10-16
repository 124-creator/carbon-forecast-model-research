"""Finite Stage F experiment. Stops before market when synthetic controls fail.
No downloads, no old prediction arrays, and no 2022+ market scoring.
"""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,csv,time,hashlib,sys,traceback,datetime
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from model import *
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'prior'))
from lowdim import filter_bank,profile_nll

ALPHAS=[.001,.01,.1,1.]
MODEL_SEEDS=[7,17,29]
DATA_SEEDS=list(range(300,306))
PROCESSES=['null','linear','interaction','asymmetric_negative_control']
BASE=Path(__file__).resolve().parents[1]
HASHES={'guangzhou_reported.csv':'b75b2c37e158933562d6979ae7532a6323ef9111a9836be8b75e6fe56b3d6573','hubei_reported.csv':'1b43a0a30146517ccd4b385777f510a4bcf83395df7ad77c90cdf5476000a71d','eu_reported.csv':'ff4fa943f8a86407eceb37026bddad2b295f5066c2e66757df6478111fc26b62'}

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def audit():
    rows=[]
    for name,h in HASHES.items():
        p=BASE/'data'/name;b=p.read_bytes()
        if hashlib.sha256(b).hexdigest()!=h:raise ValueError('Input hash mismatch')
        rr=list(csv.DictReader(b.decode('utf8').splitlines()))
        ds=[datetime.date.fromisoformat(r['date']) for r in rr];ps=np.array([float(r['author_reported_price']) for r in rr])
        if ds!=sorted(ds) or len(ds)!=len(set(ds)) or not np.isfinite(ps).all():raise ValueError('Input audit')
        rows.append(dict(name=name,rows=len(rr),sha256=h,min_date=str(ds[0]),max_date=str(ds[-1]),weekends=sum(d.weekday()>4 for d in ds),missing_prices=int(np.isnan(ps).sum()),nonpositive=int((ps<=0).sum()),model_columns=['date','author_reported_price']))
    return rows

def load_market(cutoff='2021-12-31'):
    # Never returns row metadata or later prices to any predictor or score function.
    with open(BASE/'data/guangzhou_reported.csv') as f:rr=list(csv.DictReader(f))
    rr=[r for r in rr if r['date']<=cutoff]
    return np.array([r['date'] for r in rr]),np.array([float(r['author_reported_price']) for r in rr])

def conditional_mean(r,t,process):
    if process=='null':return 0.
    if process=='linear':return .5*r[t-1]
    if process=='interaction':return .65*r[t-1]*np.tanh(r[t-2]*r[t-3])
    if process=='asymmetric_negative_control':return .5*r[t-1]+.35*np.tanh(r[t-2]**2-1)
    raise ValueError(process)

def generate(process,seed,n=1800,burn=300):
    rng=np.random.default_rng(seed);r=np.zeros(n+burn+3);mu=np.zeros_like(r)
    for t in range(3,len(r)):
        mu[t]=conditional_mean(r,t,process);r[t]=mu[t]+rng.normal()
    r=r[-n:];mu=mu[-n:]
    return 100.+np.cumsum(r),mu

def configs():
    cs=[]
    for kind in ['INTERACTION','MLP','ODD_MLP','FROZEN','LINEAR8','LINEAR60']:
        for a in ALPHAS:cs.append(dict(family=kind,alpha=a))
    for fam in ['KRR','ODD_KRR']:
        for b in [.5,2.,8.]:
            for a in [.01,.1,1.]:cs.append(dict(family=fam,band=b,alpha=a))
    for le in [7,15]:
        for lr in [.05,.1]:
            for reg in [.1,1.]:cs.append(dict(family='TREE',leaves=le,lr=lr,l2=reg))
    for fam in ['CORE_ML','DRIFT']:
        for w in [120,240,480]:cs.append(dict(family=fam,window=w))
    return [dict(family='RW')]+cs

def cname(c):return c['family']+'_'+'_'.join(f'{k}{v}' for k,v in c.items() if k!='family')

def nn_seeds(family):return MODEL_SEEDS if family in ['INTERACTION','MLP','ODD_MLP','FROZEN'] else [0]

def fit_predict(p,x,loc,raw,hs,c,fit_end,origins,seed,window=None,bank=None):
    family=c['family'];idx=support(fit_end,window);Y=targets(p,idx);origins=np.asarray(origins,int)
    if origins.min()<fit_end:raise ValueError('Prediction origin before fit cutoff')
    if idx.max()+20>=fit_end:raise ValueError('Maturity violation')
    log=dict(family=family,seed=seed,fit_end_exclusive=int(fit_end),max_training_origin=int(idx.max()),max_training_target=int(idx.max()+20),n_train=int(len(idx)))
    if family=='RW':return np.zeros((len(origins),3)),log,None
    if family=='DRIFT':
        inc=np.diff(p[max(0,fit_end-c['window']-1):fit_end]).mean()
        return np.tile(H*inc,(len(origins),1)),log,None
    if family=='CORE_ML':
        bp,ll,_=bank if bank is not None else filter_bank(p)
        j=int(np.argmin(profile_nll(ll,max(1,fit_end-c['window']),fit_end)))
        log['selected_model']=j
        return bp[origins,j],log,None
    if family=='LINEAR60':
        sc=StandardScaler().fit(raw[idx]);m=Ridge(alpha=c['alpha']*len(idx),solver='cholesky')
        m.fit(sc.transform(raw[idx]),Y)
        return m.predict(sc.transform(raw[origins])),log,None
    if family in ['INTERACTION','MLP','ODD_MLP','FROZEN','LINEAR8']:
        m=fit_profile(family,x[idx],loc[idx],Y,hs,c['alpha'],seed)
        log.update(m.log);pr=m.predict(x[origins],loc[origins])
        state={'theta':m.enc.theta.detach().numpy(),'B':m.B.numpy()}
        return pr,log,state
    if family in ['KRR','ODD_KRR']:
        m=fit_kernel(x[idx],loc[idx],Y,hs,c['alpha'],c['band'],family=='ODD_KRR')
        return m.predict(x[origins],loc[origins]),log,None
    if family=='TREE':
        m=fit_tree(x[idx],loc[idx],Y,hs,c['leaves'],c['lr'],c['l2'])
        return m.predict(x[origins],loc[origins]),log,None
    raise ValueError(family)

def normalized_loss(pred,y,hs):return float(np.mean((np.asarray(pred)-y)**2/hs**2))
def relative_mse(pred,y):return np.mean((np.asarray(pred)-y)**2,axis=-2)/np.mean(y*y,axis=0)

def run_synthetic(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'archives').mkdir(exist_ok=True)
    allrows=[];selected=[];logs=[];summary={};start=time.monotonic();fit_counter=0
    cs=configs()
    dump(out/'settings.json',dict(architecture_protocol_commit='bec82a63c1bba9f029dec3270ee397af273cf7fa',model_seeds=MODEL_SEEDS,data_seeds=DATA_SEEDS,configs=cs,processes=PROCESSES,scale_initial=800,steps=160))
    for process in PROCESSES:
      for ds in DATA_SEEDS:
        dump(out/'status.json',dict(phase='synthetic',process=process,data_seed=ds,completed_paths=len(selected)//len(set(c['family'] for c in cs)),seconds=time.monotonic()-start))
        p,mu=generate(process,ds);x,loc,raw=prepare(p);hs=training_scales(p,800)
        bank=filter_bank(p);tune=np.arange(800,1180);ev=np.arange(1220,1780);yt=targets(p,tune);ye=targets(p,ev)
        arrays={'price':p,'oracle_one_step_mean':mu[ev+1],'tune_origins':tune,'eval_origins':ev};scores={};predictions={}
        for c in cs:
            name=cname(c);ps=[]
            for seed in nn_seeds(c['family']):
                pr,log,_=fit_predict(p,x,loc,raw,hs,c,800,tune,seed,bank=bank);fit_counter+=1
                ps.append(pr);log.update(process=process,data_seed=ds,phase='tune',config=name);logs.append(log)
            ps=np.stack(ps);arrays['tune_'+name]=ps
            sl=float(np.mean([normalized_loss(q,yt,hs) for q in ps]));scores[name]=sl;predictions[name]=ps
            allrows.append(dict(process=process,data_seed=ds,config=name,family=c['family'],tune_loss=sl,config_json=json.dumps(c,sort_keys=True)))
        for fam in sorted(set(c['family'] for c in cs)):
            pool=[c for c in cs if c['family']==fam];best=min(pool,key=lambda c:scores[cname(c)])
            prlist=[]
            for seed in nn_seeds(fam):
                pr,log,state=fit_predict(p,x,loc,raw,hs,best,1200,ev,seed,bank=bank);fit_counter+=1;prlist.append(pr)
                log.update(process=process,data_seed=ds,phase='evaluation',config=cname(best));logs.append(log)
                if state is not None:
                    for key,value in state.items():arrays[f'state_{fam}_{seed}_{key}']=value
            arr=np.stack(prlist);arrays['eval_'+fam]=arr
            rr=np.mean([relative_mse(a,ye) for a in arr],axis=0)
            selected.append(dict(process=process,data_seed=ds,family=fam,selected_config=best,relative_mse=float(rr.mean()),relative_mse_h=rr.tolist(),mse_h=np.mean(np.mean((arr-ye)**2,axis=1),axis=0).tolist(),mae_h=np.mean(np.mean(abs(arr-ye),axis=1),axis=0).tolist(),forecast_ensemble=False))
        np.savez_compressed(out/'archives'/f'{process}_{ds}.npz',**arrays)
        dump(out/'selected_records.json',selected);dump(out/'all_configs.json',allrows);dump(out/'fit_logs.json',logs)
        print(f'completed {process} seed={ds}; cumulative fits={fit_counter}; elapsed={time.monotonic()-start:.1f}s',flush=True)
    families=sorted(set(r['family'] for r in selected))
    for proc in PROCESSES:
        summary[proc]={f:dict(relative_mse=float(np.mean([r['relative_mse'] for r in selected if r['process']==proc and r['family']==f])),relative_mse_h=np.mean([r['relative_mse_h'] for r in selected if r['process']==proc and r['family']==f],axis=0).tolist()) for f in families}
    it=summary['interaction'];criteria=dict(null_ok=summary['null']['INTERACTION']['relative_mse']<=1.05,linear_ok=summary['linear']['INTERACTION']['relative_mse']<.99,nonlinear_signal_ok=it['INTERACTION']['relative_mse_h'][0]<=.95,beats_linear8_on_nonlinear=it['INTERACTION']['relative_mse_h'][0]<it['LINEAR8']['relative_mse_h'][0],finite=all(np.isfinite(r['relative_mse']) for r in selected))
    res=dict(summary=summary,minimum_criteria=criteria,minimum_gate=all(criteria.values()),unique_generated_paths=len(PROCESSES)*len(DATA_SEEDS),configuration_records=len(allrows),selected_records=len(selected),fit_requests=fit_counter,seconds=time.monotonic()-start,market_executed=False,scientific_status='SYNTHETIC_MINIMUM_PASSED_NOT_NOVELTY' if all(criteria.values()) else 'STOP_SYNTHETIC_MINIMUM_FAILED')
    dump(out/'RESULTS.json',res);dump(out/'status.json',dict(phase='completed',seconds=time.monotonic()-start,status=res['scientific_status']))
    print(json.dumps(res,indent=2),flush=True)
    return res

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out)
    if out.exists() and list(out.iterdir()):raise FileExistsError('Refuse to overwrite existing run')
    out.mkdir(parents=True,exist_ok=True);dump(out/'data_audit.json',audit())
    with threadpool_limits(limits=1):run_synthetic(out)
