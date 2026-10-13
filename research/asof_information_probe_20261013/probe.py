"""Synthetic-only delay mechanism probe; no real-market information gain claim."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import csv,json,time,hashlib,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
H=np.array([1,5,20])

def delay_filter(increments,events,rho,q,beta,noise,mode='aware',max_delay=8):
    n=len(increments);dim=max_delay+2
    F=np.zeros((dim,dim));F[0,0]=rho
    for i in range(1,dim):F[i,i-1]=1
    Q=np.zeros((dim,dim));Q[0,0]=q
    m=np.zeros(dim);P=np.eye(dim)*q/(1-rho*rho)
    by={}
    for release,source,value,r in events:
        if release<source or release-source>max_delay:raise ValueError('invalid release age')
        by.setdefault(release,[]).append((source,value,r))
    out=np.zeros(n)
    for t in range(n):
        m=F@m;P=F@P@F.T+Q
        observations=[]
        if t>0 and beta!=0:observations.append((1,increments[t]/beta,noise/(beta*beta)))
        for s,v,r in by.get(t,[]):observations.append((0 if mode=='naive' else t-s,v,.5 if mode=='fixedR' else r))
        for j,v,r in observations:
            cross=P[:,j].copy();den=P[j,j]+r;gain=cross/den
            m+=gain*(v-m[j]);P-=np.outer(cross,cross)/den;P=(P+P.T)/2
        out[t]=m[0]
    return out

def load_axis():
    path=ROOT/'data/parsed/figshare_28682348/guangzhou_reported.csv'
    with path.open('rb',buffering=0) as f:lines=[f.readline() for _ in range(951)];size=f.tell()
    rows=list(csv.DictReader(b''.join(lines).decode('utf-8-sig').splitlines()))
    dates=np.array([r['date'] for r in rows])
    if len(rows)!=950 or dates[-1]!='2021-12-31' or np.any(dates[1:]<=dates[:-1]):raise ValueError('prefix violation')
    origins=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
    if len(origins)!=451:raise ValueError('451 protocol violation')
    return dates,origins,dict(source=str(path),rows=950,bytes_consumed=size,prefix_sha256=hashlib.sha256(b''.join(lines)).hexdigest(),boundary_requested=False,price_conversion=False,disclosure='950 permitted records decoded including their price text; prices never converted or used. No boundary record requested. Prior verified record count dependency.')

def generate(n,seed,scenario):
    rng=np.random.default_rng(seed);rho=.95;q=.2;beta=0. if scenario=='null' else 1.
    z=np.zeros(n);inc=np.zeros(n);events=[]
    for t in range(n):
        z[t]=rho*(z[t-1] if t else 0)+rng.normal(0,np.sqrt(q))
        if t:inc[t]=beta*z[t-1]+rng.normal(0,np.sqrt(.8))
        delay=0 if scenario=='zero_delay' else int(rng.choice([0,2,8]))
        r=float(rng.choice([.05,4.]))
        events.append((t+delay,t,z[t]+rng.normal(0,np.sqrt(r)),r))
    return np.cumsum(inc),inc,z,events,beta

def features(inc,events,beta,mode,rho):
    n=len(inc);x=np.zeros((n,5))
    for t in range(5,n):x[t]=inc[t-4:t+1][::-1]
    if mode=='price_only':return x
    if mode in ['hold','hold_age']:
        v=np.zeros(n);ages=np.zeros(n);variances=np.ones(n)*4.;latest=-1;quality=4.;by={}
        for release,source,value,r in events:by.setdefault(release,[]).append((source,value,r))
        for t in range(n):
            v[t]=v[t-1] if t else 0
            for s,a,r in by.get(t,[]):
                if s>latest:latest=s;v[t]=a;quality=r
            ages[t]=t-latest if latest>=0 else t+1;variances[t]=quality
        if mode=='hold_age':return np.column_stack([x,v,ages,variances])
    else:v=delay_filter(inc,events,rho,.2,beta,.8,mode)
    return np.column_stack([x,v])

def fit_predict(p,x,t,origins,alpha):
    out=[];tr=[]
    for h in H:
        ids=np.arange(max(60,t-480),t-int(h))
        a=x[ids];y=p[ids+h]-p[ids];mu=a.mean(0);sd=a.std(0);sd[sd==0]=1
        a=(a-mu)/sd;ym=y.mean();g=a.T@a+len(ids)*alpha*np.eye(a.shape[1]);rhs=a.T@(y-ym)
        b=np.linalg.solve(g,rhs);out.append((x[origins]-mu)/sd@b+ym)
        tr.append(dict(h=int(h),n=len(ids),max_endpoint=int(ids[-1]+h),fit_end_exclusive=int(t),normal_residual=float(np.linalg.norm(g@b-rhs)/max(1,np.linalg.norm(rhs)))))
    return np.stack(out,axis=1),tr

def score(pred,target,dates):
    cells=[]
    for year in ['2020','2021']:
        mask=np.char.startswith(dates,year)
        for j,h in enumerate(H):
            mse=float(np.mean((pred[mask,j]-target[mask,j])**2));rw=float(np.mean(target[mask,j]**2))
            cells.append(dict(year=year,h=int(h),n=int(mask.sum()),mse=mse,rw_mse=rw,ratio=mse/rw))
    return dict(score=float(np.mean([c['ratio'] for c in cells])),cells=cells)

def main():
    start=time.monotonic();dates,origins,audit=load_axis();results=[];traces=[];checks=0
    config=dict(scope='SYNTHETIC ONLY',seeds=list(range(10)),scenarios=['variable_delay','zero_delay','null'],horizons=H.tolist(),origins=451,initial=459,refit=40,window=480,alpha_grid=[.01,1.],rho_grid=[.8,.95],selection='synthetic internal pre-2020 origins 300:439, endpoints below459',budget_seconds=1200,structural_oracle='q=.2 beta and observation noise known from simulator; rho validation selected. Not learned market calibration.',modes=['price_only','hold','hold_age','aware','fixedR','naive'])
    (HERE/'CONFIG.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for scenario in config['scenarios']:
        for seed in config['seeds']:
            p,inc,z,events,beta=generate(950,seed,scenario);target=p[origins[:,None]+H]-p[origins,None]
            saved=dict(origins=origins,dates=dates[origins],target=target,synthetic_price=p,latent=z,events=np.array(events),increments=inc)
            for mode in config['modes']:
                candidates=[]
                for rho in ([.95] if mode in ['price_only','hold','hold_age'] else config['rho_grid']):
                    x=features(inc,events,beta,mode,rho)
                    for alpha in config['alpha_grid']:
                        val=np.arange(300,439);vp=np.zeros((len(val),3))
                        for end in range(300,439,40):
                            pos=np.where((val>=end)&(val<end+40))[0];vp[pos],tr=fit_predict(p,x,end,val[pos],alpha)
                            traces.extend([dict(scenario=scenario,seed=seed,mode=mode,phase='selection',rho=rho,alpha=alpha,**r) for r in tr])
                        vy=p[val[:,None]+H]-p[val,None];loss=float(np.mean(np.mean((vp-vy)**2,axis=0)/np.mean(vy**2,axis=0)))
                        candidates.append(dict(rho=rho,alpha=alpha,validation=loss))
                selected=min(candidates,key=lambda a:a['validation']);x=features(inc,events,beta,mode,selected['rho']);pred=np.zeros(target.shape)
                for end in range(459,950,40):
                    if time.monotonic()-start>1200:raise TimeoutError('budget')
                    pos=np.where((origins>=end)&(origins<end+40))[0]
                    if not len(pos):continue
                    pr,tr=fit_predict(p,x,end,origins[pos],selected['alpha']);pred[pos]=pr
                    traces.extend([dict(scenario=scenario,seed=seed,mode=mode,phase='development',**selected,**r) for r in tr])
                    q=p.copy();q[end:]+=100000
                    qp,_=fit_predict(q,x,end,origins[pos],selected['alpha'])
                    if not np.array_equal(qp,pr):raise ValueError('future label leakage')
                    checks+=1
                t=600;ev=[e for e in events if e[0]<=t]
                prefix=features(inc[:t+1],ev,beta,mode,selected['rho'])
                if not np.array_equal(prefix[-1],x[t]):raise ValueError('release leakage')
                saved[mode]=pred;s=score(pred,target,dates[origins]);results.append(dict(scenario=scenario,seed=seed,mode=mode,candidates=candidates,selected=selected,**s))
            saved['random_walk']=np.zeros(target.shape)
            np.savez_compressed(HERE/f'synthetic_{scenario}_{seed}.npz',**saved)
    aggregate=[]
    for scenario in config['scenarios']:
        for mode in config['modes']:
            v=[r['score'] for r in results if r['scenario']==scenario and r['mode']==mode]
            aggregate.append(dict(scenario=scenario,mode=mode,mean=float(np.mean(v)),sd=float(np.std(v,ddof=1))))
    for name,obj in [('INPUT_AUDIT.json',audit),('ALL_RESULTS.json',results),('AGGREGATE.json',aggregate),('TRAINING_TRACES.json',traces),('RUN_RECEIPT.json',dict(seconds=time.monotonic()-start,future_label_checks=checks,scenario_seed_runs=30,configs=len(results),max_normal_residual=max(r['normal_residual'] for r in traces),numpy=np.__version__,python=sys.version,no_market_performance=True))]:
        (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(aggregate,indent=2))

def verify():
    dates,origins,audit=load_axis();results=json.loads((HERE/'ALL_RESULTS.json').read_text(encoding='utf-8'));errors=[];count=0
    for r in results:
        with np.load(HERE/f"synthetic_{r['scenario']}_{r['seed']}.npz",allow_pickle=False) as f:
            if not np.array_equal(f['origins'],origins):raise ValueError('origin mismatch')
            target=f['synthetic_price'][origins[:,None]+H]-f['synthetic_price'][origins,None]
            if not np.array_equal(target,f['target']):raise ValueError('target mismatch')
            s=score(f[r['mode']],target,f['dates']);errors.append(abs(s['score']-r['score']));count+=1
    if count!=180 or max(errors)>1e-12:raise ValueError('verification')
    v=dict(configs=count,origins=451,max_score_error=max(errors),optimized=not __debug__,scope='saved score replay; not independent retraining',prefix=audit)
    (HERE/('VERIFY_OPTIMIZED.json' if not __debug__ else 'VERIFY.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(v,ensure_ascii=False))
if __name__=='__main__':verify() if '--verify' in sys.argv else main()
