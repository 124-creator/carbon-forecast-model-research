"""Independent NumPy nested finite-difference prototype. No market reader."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).parent
H=np.array([1,5,20]); EPS=2e-5; ALPHA=.01

def grad(f,v,eps=EPS):
    out=np.empty(len(v))
    for i in range(len(v)):
        d=np.zeros(len(v)); d[i]=eps
        out[i]=(f(v+d)-f(v-d))/(2*eps)
    return out

def generate(kind,seed):
    r=np.random.default_rng(seed); n=600; d=np.zeros(n); state=.2
    noise=r.normal(0,.08,n)
    for t in range(1,n):
        regime=1 if (t//100)%2==0 else -1
        if kind=='nonlinear_shift': state=.70*state+regime*.30*np.tanh(2*state)+noise[t]; d[t]=state+.20*np.tanh(3*state)
        elif kind=='linear': state=.8*state+noise[t]; d[t]=state
        else: d[t]=noise[t]
    scale=np.std(d[:70]); p=np.cumsum(d)/scale
    x=np.zeros((n,6))
    for t in range(6,n): x[t]=d[t-5:t+1]/scale
    return x,p

def task(start):
    a=np.arange(start,start+32); b=np.arange(start+33,start+41); q=np.arange(start+61,start+69)
    if not (max(a)+1<min(b) and max(b)+20<min(q)): raise AssertionError('maturity')
    return a,b,q

def encode(theta,x): return np.tanh(x@theta[:12].reshape(2,6).T+theta[12:14])
def fit(theta,x,ids):
    z=encode(theta,x[ids]); nxt=encode(theta,x[ids+1]); gram=z.T@z+1e-4*np.eye(2)
    return np.linalg.solve(gram,z.T@nxt)
def predict(v,x,ids,k):
    z=encode(v[:14],x[ids]); accum=np.zeros(len(ids)); out=[]
    for j in range(1,21):
        z=z@k; accum+=z@v[14:]
        if j in H: out.append(accum.copy())
    return np.array(out).T

def loss(v,x,p,ids,k):
    pred=predict(v,x,ids,k); y=p[ids[:,None]+H]-p[ids,None]
    return float(np.mean((pred-y)**2))
def adapted(v,x,p,a,b,mode):
    if mode in ('candidate','pretrain_adapt','non_tail'):
        # All 14 encoder coordinates change; readout is deliberately unchanged.
        support=a[:8] if mode=='non_tail' else b
        f=lambda theta: loss(np.r_[theta,v[14:]],x,p,support,fit(theta,x,a))
        return np.r_[v[:14]-ALPHA*grad(f,v[:14]),v[14:]]
    return v.copy()
def objective(v,x,p,tasks,mode):
    vals=[]
    for a,b,q in tasks:
        u=adapted(v,x,p,a,b,mode)
        vals.append(loss(u,x,p,q,fit(u[:14],x,np.r_[a,b])))
    return float(np.mean(vals))

def verify():
    x,p=generate('nonlinear_shift',11); a,b,q=task(75)
    v=np.random.default_rng(11).normal(0,.15,16); direction=np.random.default_rng(3).normal(size=16); direction/=np.linalg.norm(direction)
    f=lambda w: objective(w,x,p,[(a,b,q)],'candidate')
    t=time.perf_counter(); value=f(v); g=grad(f,v); elapsed=time.perf_counter()-t
    directional=(f(v+1e-4*direction)-f(v-1e-4*direction))/2e-4
    outer_error=abs(g@direction-directional)/max(1,abs(directional))
    inner=lambda w:loss(np.r_[w,v[14:]],x,p,b,fit(w,x,a))
    gi=grad(inner,v[:14]); gi2=grad(inner,v[:14],EPS/2)
    inner_error=float(np.max(np.abs(gi-gi2)))
    u=adapted(v,x,p,a,b,'candidate'); original=predict(u,x,q,fit(u[:14],x,np.r_[a,b]))
    xx=x.copy(); pp=p.copy(); cutoff=int(q[0]); xx[cutoff+1:]+=999; pp[cutoff+1:]+=999
    uu=adapted(v,xx,pp,a,b,'candidate'); changed=predict(uu,xx,np.array([cutoff]),fit(uu[:14],xx,np.r_[a,b]))
    future_error=float(np.max(abs(original[0]-changed[0])))
    prefix=adapted(v,x[:cutoff+1],p[:cutoff+1],a,b,'candidate')
    if outer_error>1e-3 or inner_error>1e-3 or future_error>1e-10 or not np.allclose(prefix,u): raise AssertionError('derivative/future/prefix validation failed')
    if np.allclose(u[:14],v[:14]) or not np.array_equal(u[14:],v[14:]): raise AssertionError('adaptation identity')
    return dict(single_task_value=value,single_task_value_plus_outer_gradient_seconds=elapsed,outer_relative_error=float(outer_error),inner_gradient_error=inner_error,future_error=future_error,prefix_replay=True,encoder_adaptation_norm=float(np.linalg.norm(u[:14]-v[:14])))

def run():
    started=time.perf_counter(); protocol=ROOT/'PROTOCOL.json'; cfg=json.loads(protocol.read_text()); deadline=started+cfg['budget_seconds']
    checks=verify(); (ROOT/'checks.json').write_text(json.dumps(checks,indent=2)); rows=[]
    for kind in cfg['processes']:
      for seed in cfg['seeds']:
        x,p=generate(kind,seed); initial=np.random.default_rng(seed).normal(0,.15,16)
        train=[task(75),task(175)]; tests=[task(325),task(425)]
        for mode in cfg['methods']:
          begin=time.perf_counter(); calls=[0]; trace=[]; best=[float('inf'),initial.copy()]
          train_mode='no_adaptation' if mode=='pretrain_adapt' else mode
          def f(w):
            if time.perf_counter()>deadline: raise TimeoutError('frozen 20 minute budget')
            if calls[0]>=180: raise RuntimeError('matched objective evaluation budget exhausted')
            calls[0]+=1
            z=objective(w,x,p,train,train_mode); trace.append(z)
            if z<best[0]: best[:]=[z,w.copy()]
            return z
          v=initial.copy(); status='not_optimized'; success=True; message=''; nit=0
          try:
            if mode not in ('RW','fixed_dictionary'):
                try:
                    result=minimize(f,v,method='L-BFGS-B',jac=lambda w:grad(f,w),options={'maxiter':8,'maxfun':180,'gtol':1e-5,'ftol':1e-9})
                    v=result.x; success=bool(result.success); message=str(result.message); nit=int(result.nit); status='optimized'
                except RuntimeError as e:
                    if str(e)!='matched objective evaluation budget exhausted': raise
                    v=best[1]; success=False; message=str(e); status='budget_exhausted_best_evaluated_point'
            elif mode=='fixed_dictionary':
                # Same frozen encoder; optimize two readout parameters, no adaptation.
                result=minimize(lambda c:f(np.r_[initial[:14],c]),v[14:],method='L-BFGS-B',options={'maxiter':8,'maxfun':180}); v[14:]=result.x; success=bool(result.success); message=str(result.message); nit=int(result.nit)
            details=[]; losses=[]
            for a,b,q in tests:
                u=adapted(v,x,p,a,b,mode); k=fit(u[:14],x,np.r_[a,b]); pred=np.zeros((len(q),3)) if mode=='RW' else predict(u,x,q,k)
                y=p[q[:,None]+H]-p[q,None]; mse=np.mean((pred-y)**2,axis=0); losses.append(float(np.mean(mse)))
                z=encode(u[:14],x[np.r_[a,b]]); sv=np.linalg.svd(z-z.mean(0),compute_uv=False)
                details.append(dict(origins=q.tolist(),targets=(q[:,None]+H).tolist(),predictions=pred.tolist(),actual=y.tolist(),mse_by_h=mse.tolist(),theta=u[:14].tolist(),readout=u[14:].tolist(),K=k.tolist(),spectral_radius=float(max(abs(np.linalg.eigvals(k)))),gram_condition=float(np.linalg.cond(z.T@z+1e-4*np.eye(2))),centered_singular_values=sv.tolist(),collapse=bool(sv[-1]<1e-4 or sv[-1]/max(sv[0],1e-12)<1e-3),spectral_explosion=bool(max(abs(np.linalg.eigvals(k)))>1.2),support_max_target=int(max(b)+20),query_min_origin=int(min(q))))
            row=dict(process=kind,seed=seed,method=mode,test_loss=float(np.mean(losses)),success=success,message=message,nit=nit,status=status,objective_calls=calls[0],runtime_seconds=time.perf_counter()-begin,training_trace=trace,parameters=v.tolist(),details=details)
          except Exception as e: row=dict(process=kind,seed=seed,method=mode,success=False,error=repr(e),runtime_seconds=time.perf_counter()-begin,training_trace=trace,objective_calls=calls[0])
          rows.append(row); (ROOT/'results.json').write_text(json.dumps(rows,indent=2)); print(kind,seed,mode,row.get('test_loss'),row['success'],flush=True)
    passed=True
    for seed in cfg['seeds']:
        group={r['method']:r for r in rows if r['seed']==seed and r['process']=='nonlinear_shift'}; c=group['candidate']
        passed &= c['success'] and 'test_loss' in c and all(c['test_loss']<group[m].get('test_loss',-np.inf) for m in ['no_adaptation','pretrain_adapt','non_tail','RW']) and all(not d['collapse'] and not d['spectral_explosion'] for d in c.get('details',[]))
    summary=dict(gate_pass=bool(passed),market_opened=False,elapsed_seconds=time.perf_counter()-started,protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),count=len(rows),checks=checks,limitation='caps match; actual nested objective work differs; maxfun may be exceeded by one numerical gradient batch')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': run()
