"""Final bounded synthetic-only stability repair. No historical writes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,hashlib,inspect,warnings
from pathlib import Path
import numpy as np
import autograd.numpy as an
from autograd import grad,value_and_grad
from autograd.extend import primitive,defvjp
from scipy.optimize import minimize
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent))
import audit as a
@primitive
def stable_tanh(x): return np.tanh(x)
# Recompute through the same primitive, not a detached cached numpy answer:
# higher-order VJPs must differentiate tanh again.
defvjp(stable_tanh,lambda ans,x:lambda g:g*(1-stable_tanh(x)**2))
def encode(t,x): return stable_tanh(an.dot(x,an.reshape(t[:12],(2,6)).T)+t[12:14])
a.encode=encode
CFG={'seeds':[11,29],'budget_seconds':600,'positive_fit_seconds':120,'other_fit_seconds':9,'maxiter':1200,'maxfun':3000,'gradient_inf_gate':1e-4,'positive_gate':'both ordinary seeds scipy success, fresh gradient <=1e-4, test MSE <1% RW; oracle gates unchanged','repair':'local recursive primitive VJP g*(1-tanh(x)^2), shared by every method; mathematical equivalence, not innovation','constraint':None,'reason_no_constraint':'No new identification constraint: final experiment tests numerical repair only; if positive gate fails stop without a second training round.','methods':['candidate','no_adaptation','pretrain_adapt','non_tail','fixed_dictionary','RW'],'processes':['nonlinear_shift','linear','no_signal'],'market_opened':False,'posthoc':True}
def dump(name,v): (R/name).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def checks():
 vals=np.array([-1000.,-400.,-20.,-2.,0.,2.,20.,400.,1000.])
 f=lambda x:an.sum(stable_tanh(x))
 g=grad(f)(vals); h=grad(lambda x:an.sum(grad(f)(x)))(vals); third=grad(lambda x:an.sum(grad(lambda y:an.sum(grad(f)(y)))(x)))(vals)
 t=np.tanh(vals); require(np.all(np.isfinite([g,h,third])),'nonfinite high-order tanh')
 require(np.allclose(g,1-t*t) and np.allclose(h,-2*t*(1-t*t)) and np.allclose(third,(-2+6*t*t)*(1-t*t)),'analytic derivative mismatch')
 x,p=a.old.generate('nonlinear_shift',11); v=np.random.default_rng(11).normal(0,.15,16); tasks=[a.old.task(75)]
 f=lambda w:a.objective(w,x,p,tasks,'candidate'); gg=grad(f)(v); d=np.random.default_rng(29).normal(size=16); d/=np.linalg.norm(d); eps=1e-5
 fd=float((f(v+eps*d)-f(v-eps*d))/(2*eps)); ad=float(gg@d); require(abs(fd-ad)/max(1,abs(fd))<1e-4,'directional gradient mismatch')
 require(float(f(v-1e-6*gg))<float(f(v)),'negative gradient does not descend')
 return dict(high_order_finite=True,analytic_orders_verified=3,directional_fd=fd,directional_ad=ad,negative_gradient_descends=True,parent_equivalence=a.checks())
def train(kind,seed,mode,x,p,deadline):
 begin=time.perf_counter(); limit=min(deadline,begin+(120 if kind=='positive_2d' else 9)); initial=np.random.default_rng(seed).normal(0,.15,16); restricted=mode=='fixed_dictionary'; init=initial[14:] if restricted else initial
 expand=lambda w:an.concatenate((initial[:14],w)) if restricted else w
 tm='no_adaptation' if mode=='pretrain_adapt' else mode
 f=lambda w:a.objective(expand(w),x,p,[a.old.task(75),a.old.task(175)],tm)
 vg=value_and_grad(f); accepted=init.copy(); steps=[]; ev=[]; success=False; message='RW'
 def fun(w):
  if time.perf_counter()>=limit: raise TimeoutError('frozen per-fit/global budget')
  val,g=vg(w)
  if not np.isfinite(val) or not np.all(np.isfinite(g)): raise FloatingPointError('nonfinite objective/gradient')
  ev.append(dict(value=float(val),gradient_inf=float(np.max(abs(g)))))
  return float(val),np.array(g)
 def cb(w):
  nonlocal accepted
  accepted=w.copy(); steps.append(w.tolist())
 if mode!='RW':
  try:
   result=minimize(fun,init,jac=True,method='L-BFGS-B',callback=cb,options=dict(maxiter=1200,maxfun=3000,gtol=1e-5,ftol=1e-12)); accepted=result.x; success=bool(result.success); message=str(result.message)
  except (TimeoutError,FloatingPointError) as e: message=str(e)
 gn=None if mode=='RW' else float(np.max(abs(grad(f)(accepted))))
 v=np.array(expand(accepted)); ds=a.details(v,x,p,mode)
 return dict(process=kind,seed=seed,method=mode,success=success,message=message,gradient_inf=gn,sufficiently_converged=bool(success and gn is not None and np.isfinite(gn) and gn<=1e-4),accepted_steps=len(steps),accepted_parameters=steps,value_gradient_calls=len(ev),training_evaluations=ev,parameters=v.tolist(),details=ds,test_loss=float(np.mean([d['mse'] for d in ds])),runtime_seconds=time.perf_counter()-begin)
def main():
 start=time.perf_counter(); deadline=start+600
 protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.parent.parent.rglob('*') if p.is_file() and R not in p.parents and '__pycache__' not in p.parts}
 dump('PROTOCOL.json',CFG); dump('old_hashes.json',protected)
 ck=checks(); dump('checks.json',ck); rows=[]; oracles=[]
 for seed in CFG['seeds']:
  x,p,v,k=a.positive(seed); q=np.r_[a.old.task(325)[2],a.old.task(425)[2]]; y=p[q[:,None]+a.old.H]-p[q,None]
  oracles.append(dict(seed=seed,exact_mse=float(np.mean((a.predict(v,x,q,k)-y)**2)),fitted_operator_mse=float(np.mean([d['mse'] for d in a.details(v,x,p,'no_adaptation')]))))
  for mode in ['no_adaptation','RW']:
   row=train('positive_2d',seed,mode,x,p,deadline); rows.append(row); dump('results.json',rows); print('RESULT',seed,mode,row['gradient_inf'],row['test_loss'],row['sufficiently_converged'],row['message'],flush=True)
 gate=all(o['exact_mse']<1e-20 and o['fitted_operator_mse']<1e-5 for o in oracles)
 for seed in CFG['seeds']:
  group={r['method']:r for r in rows if r['seed']==seed}; gate=gate and group['no_adaptation']['sufficiently_converged'] and group['no_adaptation']['test_loss']<.01*group['RW']['test_loss']
 if gate:
  for kind in CFG['processes']:
   for seed in CFG['seeds']:
    x,p=a.old.generate(kind,seed)
    for mode in CFG['methods']:
     require(time.perf_counter()<deadline,'global budget exhausted')
     row=train(kind,seed,mode,x,p,deadline); rows.append(row); dump('results.json',rows); print('RESULT',kind,seed,mode,row['gradient_inf'],row['test_loss'],row['sufficiently_converged'],flush=True)
 dump('positive_oracles.json',oracles)
 unchanged=all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in protected.items()); require(unchanged,'historical file changed')
 assessable=gate and all(r['sufficiently_converged'] and not any(d['collapse'] for d in r['details']) for r in rows if r['process']=='nonlinear_shift' and r['method'] in ['candidate','no_adaptation','pretrain_adapt','non_tail'])
 summary=dict(positive_gate=bool(gate),increment_assessable=bool(assessable),decision='STOP_CURRENT_IMPLEMENTATION' if not assessable else 'CONVERGED_COMPARISONS_REQUIRE_PER_SEED_REVIEW',candidate_skipped=not bool(gate),rows=len(rows),elapsed_seconds=time.perf_counter()-start,old_artifacts_unchanged=unchanged,market_opened=False,protocol_sha256=hashlib.sha256((R/'PROTOCOL.json').read_bytes()).hexdigest(),parent_verification_commands=['python optimization_audit/verify.py','python -O optimization_audit/verify.py','python verify_parent.py'],concrete_defects=['Forecast-only loss does not identify a full-rank latent basis.','Joint encoder/operator/readout optimization has poor conditioning and must learn known expressible 2D dynamics before meta adaptation is evaluated.','Any future identification constraint must be justified separately and shared with non-meta and ordinary-adaptation controls; it is not innovation.'])
 dump('summary.json',summary); print(json.dumps(summary),flush=True)
def verify():
 rows=json.loads((R/'results.json').read_text()); s=json.loads((R/'summary.json').read_text()); expected=40 if s['positive_gate'] else 4; require(len(rows)==expected,'row count')
 for r in rows:
  x,p=a.positive(r['seed'])[:2] if r['process']=='positive_2d' else a.old.generate(r['process'],r['seed']); v=np.array(r['parameters']); ds=a.details(v,x,p,r['method']); require(np.isclose(np.mean([d['mse'] for d in ds]),r['test_loss'],rtol=1e-10,atol=1e-10),'readback metric')
  require(all(np.allclose(d['predictions'],saved['predictions'],rtol=1e-10,atol=1e-10) for d,saved in zip(ds,r['details'])),'readback predictions')
  if r['method']!='RW':
   tm='no_adaptation' if r['method']=='pretrain_adapt' else r['method']; initial=np.random.default_rng(r['seed']).normal(0,.15,16)
   expand=lambda w:an.concatenate((initial[:14],w)) if r['method']=='fixed_dictionary' else w
   w=v[14:] if r['method']=='fixed_dictionary' else v
   gn=float(np.max(abs(grad(lambda u:a.objective(expand(u),x,p,[a.old.task(75),a.old.task(175)],tm))(w))))
   require(np.isclose(gn,r['gradient_inf'],rtol=1e-8,atol=1e-8),'readback gradient'); require(r['sufficiently_converged']==bool(r['success'] and gn<=1e-4),'convergence')
 require(s['elapsed_seconds']<=600,'budget exceeded')
 require(all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in json.loads((R/'old_hashes.json').read_text()).items()),'historical hashes')
 out=dict(verified_rows=len(rows),fresh_gradient_and_prediction_readback=True,old_hashes_unchanged=True,optimized_python=not __debug__); dump('verification_optimized.json' if not __debug__ else 'verification.json',out); print(json.dumps(out))
if __name__=='__main__':
 warnings.filterwarnings('error',category=RuntimeWarning)
 if '--verify' in sys.argv: verify()
 else: main()
