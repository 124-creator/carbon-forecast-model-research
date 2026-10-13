"""Pure synthetic bounded experiment; deployment API never receives truth.
No market input files. All labels are one-step increments mature at origin+1.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=['rw','asof_ridge','fixed_reliability','estimated_kalman','joint_raw','joint_gate']

def mature_indices(release,t,window):
    return np.flatnonzero((release<=t)&(np.arange(len(release))+1<=t))[-window:]

def gate(pred,y):
    gains=y*y-(y-pred)**2
    mean=float(gains.mean())
    lower=mean-1.645*float(gains.std(ddof=1))/np.sqrt(len(y))
    return float(np.clip(lower/(mean+1e-12),0,1)) if lower>0 else 0.

def moments(x):
    v=float(np.clip(np.mean(x[:,0]*x[:,1]),.02,20))
    r=np.clip(np.mean(x*x,axis=0)-v,.02,50)
    return v,r

def joint_fit(x,y,iterations=30,fixed_r=None):
    # E/M over latent common factor with two fixed unit loadings.
    # The only information inputs are released measurements and mature y.
    v,r=moments(x); beta=float(np.mean(y*x.mean(axis=1))/(v+1e-6))
    sy=max(float(np.var(y)),.02)
    if fixed_r is not None: r=np.asarray(fixed_r).copy()
    history=[]
    for _ in range(iterations):
        var=1/(1/v+np.sum(1/r)+beta*beta/sy)
        mu=var*(np.sum(x/r,axis=1)+beta*y/sy)
        zz=mu*mu+var
        beta=float(np.clip(np.mean(y*mu)/np.mean(zz),-5,5))
        sy=float(np.clip(np.mean(y*y-2*beta*y*mu+beta*beta*zz),.02,50))
        if fixed_r is None:
            r=np.clip(np.mean(x*x-2*x*mu[:,None]+zz[:,None],axis=0),.02,50)
        v=float(np.clip(np.mean(zz),.02,20))
        history.append([beta,v,float(r[0]),float(r[1]),sy])
    return dict(beta=beta,v=v,r=r,sy=sy,history=history)

def rho_estimate(x,ids,v):
    adjacent=np.diff(ids)==1
    if not adjacent.any(): return 0.
    cross=(x[1:,0]*x[:-1,1]+x[1:,1]*x[:-1,0])/2
    return float(np.clip(np.mean(cross[adjacent])/v,0,.98))

def source_at(release,t):
    ids=np.flatnonzero(release<=t)
    return int(ids[-1]) if len(ids) else -1

def static_predict(x,release,t,m,rho):
    j=source_at(release,t)
    if j<0: return 0.
    var=1/(1/m['v']+np.sum(1/m['r']))
    z=var*np.sum(x[j]/m['r'])
    return float(m['beta']*rho**(t-j)*z)

def filter_at(x,release,t,v,r,rho):
    # Reconstruct source-time causal filter using only measurements released
    # by forecast origin t. No future observations or generated state.
    z=0.; variance=v; q=max(v*(1-rho*rho),.001)
    precision=float(np.sum(1/r)); information=x[:,0]/r[0]+x[:,1]/r[1]
    for i in range(t+1):
        if i: z*=rho; variance=rho*rho*variance+q
        if release[i]<=t:
            post=1/(1/variance+precision)
            z=post*(z/variance+information[i]); variance=post
    return z

def asof_feature(x,release,t):
    j=source_at(release,t)
    return np.array([1.,0.,0.,0.]) if j<0 else np.array([1.,x[j,0],x[j,1],(t-j)/5])

def fit_models(x,release,y,t,frozen):
    ids=mature_indices(release,t,240)
    val=ids[-80:]; train=ids[:-80]
    # No training release may cross the first internal validation origin.
    train=train[release[train]<=val[0]]
    tx=x[train]; ty=y[train]
    m=joint_fit(tx,ty)
    rho=rho_estimate(tx,train,m['v'])
    if frozen is None: frozen=m['r'].copy()
    fixed=joint_fit(tx,ty,fixed_r=frozen)
    fixedrho=rho_estimate(tx,train,fixed['v'])
    v,r=moments(tx); krho=rho_estimate(tx,train,v)
    kz=np.array([filter_at(x,release,int(s),v,r,krho) for s in train])
    kb=float(np.dot(kz,ty)/(np.dot(kz,kz)+1.))
    features=np.array([asof_feature(x,release,int(s)) for s in train])
    ridge=np.linalg.solve(features.T@features+np.diag([.001,1,1,1]),features.T@ty)
    jp=np.array([static_predict(x,release,int(s),m,rho) for s in val])
    fp=np.array([static_predict(x,release,int(s),fixed,fixedrho) for s in val])
    return dict(joint=m,rho=rho,fixed=fixed,fixedrho=fixedrho,
                jgate=gate(jp,y[val]),fgate=gate(fp,y[val]),
                kalman=(v,r,krho,kb),ridge=ridge,train=train,val=val),frozen

def deploy_predict(x,release,t,model):
    jp=static_predict(x,release,t,model['joint'],model['rho'])
    fp=static_predict(x,release,t,model['fixed'],model['fixedrho'])
    v,r,rho,beta=model['kalman']
    return np.array([0.,float(asof_feature(x,release,t)@model['ridge']),
                     fp*model['fgate'],beta*filter_at(x,release,t,v,r,rho),
                     jp,jp*model['jgate']])

def generate(mode,seed,n=1200):
    rng=np.random.default_rng(seed)
    latent=np.zeros(n)
    for t in range(1,n): latent[t]=.85*latent[t-1]+rng.normal()*np.sqrt(1-.85**2)
    noise=rng.normal(size=(n,2)); error=rng.normal(size=n)
    scale=np.tile([.3,1.4],(n,1))
    if mode=='reliability_shift': scale[n//2:]=[2.5,.25]
    beta=0. if mode=='null' else 1.
    y=beta*latent+.7*error
    x=latent[:,None]+noise*scale
    release=np.arange(n)+rng.integers(0,6,n)
    # Truth returned separately; never passed to fit_models or deploy_predict.
    return (x,release,y),dict(latent=latent,beta=beta,scale=scale)

def jsonable(obj):
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,(np.integer,np.floating)): return obj.item()
    raise TypeError(type(obj).__name__)

def save_json(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=jsonable),encoding='utf-8')

def run():
    start=time.perf_counter(); rows=[]; timings={}; traces=[]
    for mode in ['signal','null','reliability_shift']:
        for seed in range(12):
            if time.perf_counter()-start>1800:
                save_json('FAILURE.json',dict(reason='budget',completed=len(rows))); return
            cellstart=time.perf_counter()
            (x,release,y),truth=generate(mode,seed)
            pred=np.full((len(y),len(METHODS)),np.nan); frozen=None; audit=[]; estimates=[]
            for t in range(300,len(y)-1):
                if t%20==0:
                    model,frozen=fit_models(x,release,y,t,frozen)
                    assert np.all(release[model['train']]<=model['val'][0])
                    assert np.all(model['train']<model['val'][0])
                    assert np.all(release[model['val']]<=t)
                    assert np.all(model['val']+1<=t)
                    audit.append(dict(t=t,train=model['train'],val=model['val']))
                    estimates.append(dict(t=t,joint=model['joint'],rho=model['rho'],
                                          jgate=model['jgate'],fgate=model['fgate'],fixed=model['fixed'],kalman=model['kalman']))
                pred[t]=deploy_predict(x,release,t,model)
                if not np.all(np.isfinite(pred[t])): raise RuntimeError('nonfinite')
                if time.perf_counter()-start>1800: raise TimeoutError('1800 second cap')
            # Diagnostic oracle evaluated only after all deployable forecasts.
            oracle=truth['beta']*truth['latent']
            np.savez_compressed(ROOT/f'pred_{mode}_{seed}.npz',x=x,release=release,y=y,pred=pred,oracle=oracle)
            mask=np.arange(600,len(y)-1)
            mse=np.mean((pred[mask]-y[mask,None])**2,axis=0)
            record=dict(mode=mode,seed=seed,mse=dict(zip(METHODS,mse)),
                        ratio=dict(zip(METHODS,mse/mse[0])),
                        oracle_diagnostic_mse=float(np.mean((oracle[mask]-y[mask])**2)),
                        segments={str(a):dict(zip(METHODS,np.mean((pred[a:b]-y[a:b,None])**2,axis=0))) for a,b in [(600,900),(900,1199)]},
                        seconds=time.perf_counter()-cellstart)
            rows.append(record); traces.append(dict(mode=mode,seed=seed,estimates=estimates,audit=audit))
            save_json('ALL_RESULTS.json',rows); save_json(f'trace_{mode}_{seed}.json',traces[-1]); traces.clear()
            print(mode,seed,record['ratio'],flush=True)
    summary={}
    for mode in ['signal','null','reliability_shift']:
        sub=[r for r in rows if r['mode']==mode]
        summary[mode]={m:dict(mean_ratio=float(np.mean([r['ratio'][m] for r in sub])),
                            sd_ratio=float(np.std([r['ratio'][m] for r in sub],ddof=1)),
                            wins_rw=sum(r['ratio'][m]<1 for r in sub),
                            pooled_mse=float(np.mean([r['mse'][m] for r in sub]))) for m in METHODS}
    save_json('SUMMARY.json',summary)
    elapsed=time.perf_counter()-start
    save_json('RECEIPT.json',dict(seconds=elapsed,cells=len(rows),cap_seconds=1800,methods=METHODS,
               reads=['no input files; generated integer-axis synthetic data only'],iterations_cap=30,search_trials=0,
               protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL_ZH.md').read_bytes()).hexdigest(),
               stop_null=summary['null']['joint_gate']['mean_ratio']>1.01,
               stop_shift=summary['reliability_shift']['joint_gate']['pooled_mse']>=summary['reliability_shift']['fixed_reliability']['pooled_mse']))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',action='store_true'); args=parser.parse_args()
    if args.run:
        try: run()
        except Exception as exc:
            save_json('FAILURE.json',dict(type=type(exc).__name__,message=str(exc))); raise
