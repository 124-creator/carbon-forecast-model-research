"""Posthoc bounded optimization audit; never reads market data or writes old results."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import autograd.numpy as an
from autograd import grad,value_and_grad
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import mechanism as old
CFG=json.loads((ROOT/'PROTOCOL.json').read_text())

def encode(t,x): return an.tanh(an.dot(x,an.reshape(t[:12],(2,6)).T)+t[12:14])
def fit(t,x,ids):
 z=encode(t,x[ids]); nxt=encode(t,x[ids+1])
 return an.linalg.solve(an.dot(z.T,z)+1e-4*an.eye(2),an.dot(z.T,nxt))
def predict(v,x,ids,k):
 z=encode(v[:14],x[ids]); total=an.zeros(len(ids)); out=[]
 for j in range(1,21):
  z=an.dot(z,k); total=total+an.dot(z,v[14:])
  if j in old.H: out.append(total)
 return an.stack(out,axis=1)
def loss(v,x,p,ids,k): return an.mean((predict(v,x,ids,k)-(p[ids[:,None]+old.H]-p[ids,None]))**2)
def adapt(v,x,p,a,b,mode):
 if mode in ('candidate','pretrain_adapt','non_tail'):
  ids=a[:8] if mode=='non_tail' else b
  f=lambda t:loss(an.concatenate((t,v[14:])),x,p,ids,fit(t,x,a))
  return an.concatenate((v[:14]-.01*grad(f)(v[:14]),v[14:]))
 return v

def objective(v,x,p,tasks,mode):
 return an.mean(an.stack([loss(adapt(v,x,p,a,b,mode),x,p,q,fit(adapt(v,x,p,a,b,mode)[:14],x,np.r_[a,b])) for a,b,q in tasks]))

def positive(seed):
 r=np.random.default_rng(seed); ang=.13; k=.999*np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
 z=np.zeros((600,2)); z[0]=r.normal(size=2); z[0]*=.55/np.linalg.norm(z[0])
 for t in range(599): z[t+1]=z[t]@k
 x=np.zeros((600,6)); x[:,:2]=np.arctanh(z)
 c=np.array([.7,-.4]); p=np.cumsum(z@c)
 theta=np.zeros(14); theta[0]=theta[7]=1
 return x,p,np.r_[theta,c],k

def details(v,x,p,mode):
 out=[]
 for a,b,q in [old.task(325),old.task(425)]:
  u=np.array(adapt(v,x,p,a,b,mode)); k=np.array(fit(u[:14],x,np.r_[a,b])); pred=np.zeros((len(q),3)) if mode=='RW' else np.array(predict(u,x,q,k)); y=p[q[:,None]+old.H]-p[q,None]
  z=np.array(encode(u[:14],x[np.r_[a,b]])); sv=np.linalg.svd(z-z.mean(0),compute_uv=False)
  out.append(dict(origins=q.tolist(),targets=(q[:,None]+old.H).tolist(),predictions=pred.tolist(),actual=y.tolist(),mse=float(np.mean((pred-y)**2)),singular_values=sv.tolist(),collapse=bool(sv[-1]<1e-4 or sv[-1]/max(sv[0],1e-12)<1e-3),spectral_radius=float(max(abs(np.linalg.eigvals(k)))),parameters=u.tolist(),K=k.tolist()))
 return out

def train(kind,seed,mode,x,p,deadline):
 initial=np.random.default_rng(seed).normal(0,.15,16); tasks=[old.task(75),old.task(175)]; trainmode='no_adaptation' if mode=='pretrain_adapt' else mode
 begin=time.perf_counter(); limit=min(deadline,begin+(60 if kind=='positive_2d' else 20)); steps=[]; evaluations=[]; calls=0
 restricted=mode=='fixed_dictionary'; init=initial[14:] if restricted else initial
 expand=lambda w:an.concatenate((initial[:14],w)) if restricted else w
 f=lambda w:objective(expand(w),x,p,tasks,trainmode)
 vg=value_and_grad(f); accepted=init.copy(); success=False; message='RW'; result=None
 def fun(w):
  nonlocal calls
  if time.perf_counter()>limit: raise TimeoutError('per-fit/global budget')
  calls+=1; val,g=vg(w); evaluations.append(dict(value=float(val),gradient_inf=float(np.max(abs(g)))))
  return float(val),np.array(g)
 def callback(w):
  nonlocal accepted
  accepted=w.copy(); steps.append(w.tolist())
 if mode!='RW':
  try:
   result=minimize(fun,init,jac=True,method='L-BFGS-B',callback=callback,options=dict(maxiter=400,maxfun=1000,gtol=1e-5,ftol=1e-12))
   accepted=result.x; success=bool(result.success); message=str(result.message)
  except TimeoutError as e: message=str(e)
 v=np.array(expand(accepted)); gn=None
 if mode!='RW': gn=float(np.max(abs(grad(f)(accepted))))
 ds=details(v,x,p,mode)
 return dict(process=kind,seed=seed,method=mode,success=success,message=message,accepted_steps=len(steps),value_gradient_calls=calls,independent_gradient_checks=0 if gn is None else 1,gradient_inf=gn,sufficiently_converged=bool(success and gn is not None and gn<=1e-4),runtime_seconds=time.perf_counter()-begin,parameters=v.tolist(),training_evaluations=evaluations,accepted_parameters=steps,details=ds,test_loss=float(np.mean([d['mse'] for d in ds])))

def original_replay():
 receipts=[]
 for seed in CFG['seeds']:
  x,p=old.generate('nonlinear_shift',seed); initial=np.random.default_rng(seed).normal(0,.15,16)
  for mode in ['candidate','no_adaptation','pretrain_adapt','non_tail']:
   calls=0; steps=[]; trainmode='no_adaptation' if mode=='pretrain_adapt' else mode
   def f(w):
    nonlocal calls
    if calls>=180: raise RuntimeError('cap')
    calls+=1; return old.objective(w,x,p,[old.task(75),old.task(175)],trainmode)
   try: minimize(f,initial,jac=lambda w:old.grad(f,w),method='L-BFGS-B',callback=lambda w:steps.append(w.tolist()),options=dict(maxiter=8,maxfun=180,gtol=1e-5,ftol=1e-9))
   except RuntimeError as e:
    if str(e)!='cap': raise
   receipts.append(dict(seed=seed,method=mode,objective_calls=calls,accepted_steps=len(steps),complete_gradient_batches=calls//33,max_outer_trial_points=calls//33+1,outer_scalar_calls_per_full_value_gradient=33,inner_loss_calls_per_candidate_outer_objective=58,old_saved_nit_is_unassigned_default=True))
 return receipts

def checks():
 x,p=old.generate('nonlinear_shift',11); v=np.random.default_rng(11).normal(0,.15,16); tasks=[old.task(75)]
 f=lambda w:objective(w,x,p,tasks,'candidate'); fo=lambda w:old.objective(w,x,p,tasks,'candidate')
 g=np.array(grad(f)(v)); gf=old.grad(fo,v); err=float(np.max(abs(g-gf))/max(1,np.max(abs(gf)))); value_error=float(abs(f(v)-fo(v)))
 if err>1e-4 or value_error>1e-4: raise RuntimeError('AD/FD equivalence failed')
 a,b,q=tasks[0]; u=np.array(adapt(v,x,p,a,b,'candidate')); pred=np.array(predict(u,x,q[:1],fit(u[:14],x,np.r_[a,b])))
 xx=x.copy(); pp=p.copy(); xx[q[0]+1:]+=999; pp[q[0]+1:]+=999
 uu=adapt(v,xx,pp,a,b,'candidate'); future=float(np.max(abs(pred-predict(uu,xx,q[:1],fit(uu[:14],xx,np.r_[a,b])))))
 if future>1e-10: raise RuntimeError('future invariance')
 # Exact rank-one witness: scalar-identifiable process cannot identify two latent coordinates.
 scalar=np.linspace(-.3,.3,50); xx=np.zeros((50,6)); xx[:,0]=scalar; t=np.zeros(14); t[0]=t[6]=1
 sv=np.linalg.svd(np.array(encode(t,xx)),compute_uv=False)
 return dict(ad_fd_relative_gradient_error=err,value_error=value_error,future_invariance_error=future,rank_one_witness_singular_values=sv.tolist(),interpretation='forecast-only loss does not enforce full-rank latent coordinates; not proof collapsed original parameters are optimal')

def run():
 start=time.perf_counter(); deadline=start+CFG['budget_seconds']; protected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.parent.iterdir() if p.is_file()}; rows=[]
 ck=checks(); replay=original_replay(); (ROOT/'original_budget_audit.json').write_text(json.dumps(replay,indent=2)); print('ORIGINAL',json.dumps(replay),flush=True)
 oracles=[]
 for seed in CFG['seeds']:
  x,p,v,k=positive(seed); q=np.r_[old.task(325)[2],old.task(425)[2]]; y=p[q[:,None]+old.H]-p[q,None]
  exact=float(np.mean((predict(v,x,q,k)-y)**2)); estimated=details(v,x,p,'no_adaptation'); fitted=float(np.mean([d['mse'] for d in estimated])); oracles.append(dict(seed=seed,exact_mse=exact,fitted_operator_mse=fitted,oracle_parameters=v.tolist(),true_K=k.tolist()))
  for mode in ['no_adaptation','RW']:
   row=train('positive_2d',seed,mode,x,p,deadline); rows.append(row); print('POSITIVE',seed,mode,row['accepted_steps'],row['test_loss'],row['gradient_inf'],row['sufficiently_converged'],flush=True)
 (ROOT/'positive_oracles.json').write_text(json.dumps(oracles,indent=2)); (ROOT/'results.json').write_text(json.dumps(rows,indent=2))
 for kind in CFG['processes']:
  for seed in CFG['seeds']:
   x,p=old.generate(kind,seed)
   for mode in CFG['methods']:
    if time.perf_counter()>deadline: raise TimeoutError('global budget reached')
    row=train(kind,seed,mode,x,p,deadline); rows.append(row); (ROOT/'results.json').write_text(json.dumps(rows,indent=2)); print(kind,seed,mode,row['accepted_steps'],row['value_gradient_calls'],row['gradient_inf'],row['sufficiently_converged'],row['test_loss'],flush=True)
 positive_gate=all(o['exact_mse']<1e-20 and o['fitted_operator_mse']<1e-5 for o in oracles)
 for seed in CFG['seeds']:
  group={r['method']:r for r in rows if r['process']=='positive_2d' and r['seed']==seed}; positive_gate &=group['no_adaptation']['sufficiently_converged'] and group['no_adaptation']['test_loss']<.01*group['RW']['test_loss']
 sufficient=all(r['sufficiently_converged'] for r in rows if r['process']=='nonlinear_shift' and r['method'] in ['candidate','no_adaptation','pretrain_adapt','non_tail'])
 unchanged=all(hashlib.sha256((ROOT.parent/n).read_bytes()).hexdigest()==h for n,h in protected.items())
 if not unchanged: raise RuntimeError('old artifacts changed')
 summary=dict(checks=ck,positive_gate=bool(positive_gate),increment_assessable=bool(sufficient),increment_verdict='not assessable without sufficient convergence' if not sufficient else 'inspect fixed-seed comparisons and latent health',count=len(rows),elapsed_seconds=time.perf_counter()-start,protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),old_artifacts_unchanged=unchanged,old_hashes=protected,market_opened=False,repair_is_existing_technique=True)
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': run()
