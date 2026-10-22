"""Causal low-dimensional price filters and bounded synthetic controls.

State model p_t=m_t+z_t, m_t=m_(t-1)+eps_t,
z_t=rho*z_(t-1)+eta_t. No smoothed states or future inputs are used.
"""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

H = np.array([1,5,20],dtype=int)

def bank_parameters():
    return np.array([(0.,1.)]+[(r,q) for r in [0.,.25,.5,.75,.9,.97] for q in [.02,.1,.25,.5,.8]])

def filter_bank(p, pars=None):
    """Return causal forecast deltas and normalized likelihood contributions.
    pars columns are rho and permanent innovation variance fraction q/(q+r).
    A common positive innovation scale is profiled out. Initial transient
    uncertainty is stationary; initial permanent level is diffuse conditional
    on the observed first total price.
    """
    p=np.asarray(p,dtype=float)
    if p.ndim!=1 or len(p)<2 or not np.isfinite(p).all():
        raise ValueError('Finite univariate prices required')
    pars=bank_parameters() if pars is None else np.asarray(pars,float)
    rho,q=pars.T; r=1-q
    if np.any((rho<0)|(rho>=1)|(q<0)|(q>1)):
        raise ValueError('Invalid stable parameters')
    n,k=len(p),len(pars)
    z=np.zeros(k); P=r/(1-rho*rho)
    pred=np.zeros((n,k,3)); ll=np.zeros((n,k,2)); states=np.zeros((n,k))
    mult=rho[:,None]**H[None,:]-1
    for t in range(1,n):
        e=p[t]-p[t-1]-(rho-1)*z
        S=(rho-1)**2*P+q+r
        C=rho*(rho-1)*P+r
        if np.any(S<=0) or not np.isfinite(S).all():raise FloatingPointError('Innovation covariance')
        K=C/S
        z=rho*z+K*e
        P=np.maximum(rho*rho*P+r-C*C/S,0)
        pred[t]=mult*z[:,None]; states[t]=z
        ll[t,:,0]=np.log(S); ll[t,:,1]=e*e/S
    return pred,ll,states

def profile_nll(ll,start,stop):
    """Profile minus twice Gaussian loglik, excluding constants; stop exclusive."""
    start=max(1,int(start)); stop=int(stop)
    if stop<=start:raise ValueError('Empty likelihood sample')
    n=stop-start
    a=ll[start:stop].sum(axis=0)
    return a[:,0]+n*np.log(np.maximum(a[:,1]/n,1e-14))

def features(p,L):
    p=np.asarray(p,float); X=np.zeros((len(p),L))
    d=np.diff(p)
    for t in range(L,len(p)):X[t]=d[t-L:t][::-1]
    return X

def targets(p,origins):
    p=np.asarray(p,float); t=np.asarray(origins,int)
    return p[t[:,None]+H]-p[t,None]

def ridge_predict(p,fit_end,origins,L,alpha):
    origins=np.asarray(origins,int)
    if fit_end<=L+20:raise ValueError('Too little matured training support')
    X=features(p,L); idx=np.arange(L,fit_end-20)
    Y=targets(p,idx); scaler=StandardScaler().fit(X[idx])
    model=Ridge(alpha=float(alpha)*len(idx),solver='cholesky')
    model.fit(scaler.transform(X[idx]),Y)
    return model.predict(scaler.transform(X[origins]))

def ratios(pred,true):
    den=np.mean(true*true,axis=0)
    return np.mean((pred-true)**2,axis=0)/den

def generate(n,seed,rho,q,r):
    rng=np.random.default_rng(seed)
    z=rng.normal()*np.sqrt(r/(1-rho*rho)) if r>0 else 0.
    m=100.; p=np.empty(n)
    for t in range(n):
        m+=np.sqrt(q)*rng.normal(); z=rho*z+np.sqrt(r)*rng.normal();p[t]=m+z
    return p

def run_controls(out):
    cfg=json.loads((Path(__file__).parent/'control_protocol.json').read_text())
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    allrows=[];start_time=time.monotonic()
    for name,par in cfg['processes'].items():
        for seed in cfg['seeds']:
            p=generate(cfg['n'],seed,**par)
            pred,ll,_=filter_bank(p)
            tune=np.arange(600,980); ev=np.arange(1000,1780)
            scale=np.mean(targets(p,np.arange(20,580))**2,axis=0)
            scores=[]
            for L in cfg['ridge_lags']:
                for a in cfg['ridge_penalty_average_loss']:
                    pr=ridge_predict(p,600,tune,L,a)
                    scores.append((float(np.mean((pr-targets(p,tune))**2/scale)),L,a))
            _,L,a=min(scores)
            ridge=ridge_predict(p,1000,ev,L,a)
            j=int(np.argmin(profile_nll(ll,1,1000)))
            oracle=filter_bank(p,np.array([[par['rho'],par['q']/(par['q']+par['r'])]]))[0][:,0]
            for method,pr in [('RW',np.zeros((len(ev),3))),('oracle',oracle[ev]),('Ridge',ridge),('lowdim_ML',pred[ev,j])]:
                rr=ratios(pr,targets(p,ev));allrows.append(dict(process=name,seed=seed,method=method,relative_mse=float(rr.mean()),by_horizon=rr.tolist(),selected_rho=float(bank_parameters()[j,0]) if method=='lowdim_ML' else None))
        print('synthetic completed:',name,flush=True)
    summary={}
    for name in cfg['processes']:
        summary[name]={m:float(np.mean([r['relative_mse'] for r in allrows if r['process']==name and r['method']==m])) for m in ['RW','oracle','Ridge','lowdim_ML']}
    rec=(1-summary['signal']['lowdim_ML'])/(1-summary['signal']['oracle'])
    gate=(summary['null']['lowdim_ML']<=1.02 and summary['signal']['lowdim_ML']<=.95 and rec>=.5)
    result=dict(summary=summary,oracle_gain_recovered=rec,gate_passed=bool(gate),seconds=time.monotonic()-start_time,rows=allrows,protocol_sha256=hashlib.sha256((Path(__file__).parent/'control_protocol.json').read_bytes()).hexdigest())
    (out/'control_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    return result

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='outputs');args=ap.parse_args();run_controls(args.out)
