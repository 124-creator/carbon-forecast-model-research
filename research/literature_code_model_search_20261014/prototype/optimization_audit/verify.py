"""Independent readback verifier: explicit raises work under python -O."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from autograd import grad
import audit as a
R=Path(__file__).resolve().parent

def require(ok,msg):
 if not ok: raise RuntimeError(msg)

def run():
 start=time.perf_counter(); rows=json.loads((R/'results.json').read_text()); summary=json.loads((R/'summary.json').read_text()); cfg=json.loads((R/'PROTOCOL.json').read_text())
 expected={(kind,seed,method) for kind in cfg['processes'] for seed in cfg['seeds'] for method in cfg['methods']}|{('positive_2d',seed,method) for seed in cfg['seeds'] for method in ['no_adaptation','RW']}
 require({(r['process'],r['seed'],r['method']) for r in rows}==expected and len(rows)==len(expected)==40,'grid completeness')
 require(summary['protocol_sha256']==hashlib.sha256((R/'PROTOCOL.json').read_bytes()).hexdigest(),'protocol hash')
 for name,h in summary['old_hashes'].items(): require(hashlib.sha256((R.parent/name).read_bytes()).hexdigest()==h,'old file changed '+name)
 checks=0; collapse={}
 for r in rows:
  kind,seed,mode=r['process'],r['seed'],r['method']; x,p=(a.positive(seed)[:2] if kind=='positive_2d' else a.old.generate(kind,seed)); v=np.array(r['parameters'])
  ds=a.details(v,x,p,mode)
  require(np.isclose(np.mean([d['mse'] for d in ds]),r['test_loss'],rtol=1e-10,atol=1e-10),'metric mismatch')
  for d,saved in zip(ds,r['details']):
   require(np.allclose(d['predictions'],saved['predictions'],rtol=1e-10,atol=1e-10),'prediction mismatch')
   require(d['collapse']==saved['collapse'],'collapse mismatch')
  collapse[str((kind,seed,mode))]=sum(d['collapse'] for d in ds)
  require(r['accepted_steps']==len(r['accepted_parameters']),'accepted step count')
  require(r['value_gradient_calls']==len(r['training_evaluations']),'call count')
  if mode!='RW':
   tm='no_adaptation' if mode=='pretrain_adapt' else mode
   if mode=='fixed_dictionary':
    initial=np.random.default_rng(seed).normal(0,.15,16)
    f=lambda c:a.objective(a.an.concatenate((initial[:14],c)),x,p,[a.old.task(75),a.old.task(175)],tm)
    gn=float(np.max(abs(grad(f)(v[14:]))))
   else: gn=float(np.max(abs(grad(lambda w:a.objective(w,x,p,[a.old.task(75),a.old.task(175)],tm))(v))))
   require(np.isclose(gn,r['gradient_inf'],rtol=1e-8,atol=1e-8),'gradient mismatch')
   require(r['sufficiently_converged']==bool(r['success'] and gn<=1e-4),'convergence mislabeled')
  checks+=1
 for o in json.loads((R/'positive_oracles.json').read_text()):
  x,p,v,k=a.positive(o['seed']); q=np.r_[a.old.task(325)[2],a.old.task(425)[2]]; y=p[q[:,None]+a.old.H]-p[q,None]
  require(np.isclose(float(np.mean((a.predict(v,x,q,k)-y)**2)),o['exact_mse'],atol=1e-28),'oracle mismatch')
 require(summary['elapsed_seconds']<900 and not summary['market_opened'],'budget/market')
 out=dict(verified_rows=checks,explicit_exception_checks=True,old_artifacts_unchanged=True,market_opened=False,collapse_task_counts=collapse,elapsed_seconds=time.perf_counter()-start)
 (R/'verification.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': run()
