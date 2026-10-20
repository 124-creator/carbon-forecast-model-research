import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import importlib.util,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent/'pilot'/'run_pilot.py'
spec=importlib.util.spec_from_file_location('pilot',PILOT)
if spec is None or spec.loader is None: raise RuntimeError('missing pilot')
pilot=importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
mask=np.concatenate([np.tile(np.arange(12)<6,20),np.tile(np.arange(12)<6,4),np.tile(np.arange(12)<6,4),np.zeros(4,dtype=bool)])
LR=.04
METHODS=['pcgrad','private_only','SP_budget','shared_eval_ablation','joint_adaptation']
def dump(n,o): (ROOT/n).write_text(json.dumps(o,indent=2),encoding='utf-8')
def obj(p,x,y): return pilot.objective(p,x,y,'shared_private')
def private_adapt(p,x,y):
 g=obj(p,x,y)[1]; q=p.copy(); q[~mask]-=LR*g[~mask]; return q
def adapt_and_gradient(p,x,y):
 q=private_adapt(p,x,y); v=obj(q,x,y)[1]; direction=v.copy(); direction[mask]=0
 eps=1e-4/max(np.linalg.norm(direction),1e-12)
 hv=(obj(p+eps*direction,x,y)[1]-obj(p-eps*direction,x,y)[1])/(2*eps)
 return q,v-LR*hv

def risks(p,x,y):
 e=(pilot.forward(p,x,'shared_private')[0]-y)**2
 return e.reshape(4,4,16).mean(2).ravel()
def candidate(p,x,y,shared_eval=False):
 q,g=adapt_and_gradient(p,x,y); z=p.copy(); z[mask]-=LR*g[mask]; r=private_adapt_from_gradient(z,x,y)
 before=risks(q,x,y); after=risks(r,x,y)
 eb=risks(p,x,y) if shared_eval else before
 ea=risks(z,x,y) if shared_eval else after
 accepted=bool(np.all(ea<=eb+1e-10))
 return (r if accepted else q),dict(accepted=accepted,before=before.tolist(),trial_after=after.tolist(),actual_after=(after if accepted else before).tolist(),gate_before=eb.tolist(),gate_after=ea.tolist())
def private_adapt_from_gradient(z,x,y):
 # This is a fifth gradient call: recompute private response at the shared trial.
 return private_adapt(z,x,y)
def run():
 start=time.monotonic(); results=[]; audits=[]
 # Candidate uses FIVE gradient evaluations; match SP accordingly.
 for seed in [17,29]:
  x,y=pilot.data(seed); np.savez_compressed(ROOT/f'data_{seed}.npz',x=x,y=y)
  for method in METHODS:
   p=pilot.init(seed); best=np.full(4,np.inf); bp=np.tile(p,(4,1)); bs=np.zeros(4,int); rng=np.random.default_rng(seed+333); ts=time.monotonic(); calls=0
   for step in range(1201):
    if time.monotonic()-start>600: raise RuntimeError('600 second total budget exceeded')
    if step%20==0:
     val=((pilot.forward(p,x[:,64:128],'shared_private')[0]-y[:,64:128])**2).mean(1); improve=val<best; best[improve]=val[improve]; bp[improve]=p; bs[improve]=step
    if step==1200: break
    xx=x[:,:64]; yy=y[:,:64]
    if method in ['joint_adaptation','shared_eval_ablation']:
     p,d=candidate(p,xx,yy,method=='shared_eval_ablation'); calls+=5; audits.append(dict(seed=seed,method=method,step=step,**d))
    elif method=='private_only': p=private_adapt(p,xx,yy); calls+=1
    elif method=='SP_budget':
     for _ in range(5): p-=LR/5*obj(p,xx,yy)[1]; calls+=1
    else:
     gs=np.array([pilot.objective(p,xx,yy,'shared_private',t)[1] for t in range(4)]); calls+=4; G=gs[:,mask]; H=G.copy()
     for i in range(4):
      for j in rng.permutation(4):
       if i==j: continue
       dot=H[i]@G[j]
       if dot<0: H[i]-=dot/(G[j]@G[j]+1e-30)*G[j]
     g=gs.mean(0); g[mask]=H.mean(0); p-=LR*g
   pred=np.array([pilot.forward(bp[t],x,'shared_private')[0][t] for t in range(4)])
   np.savez_compressed(ROOT/f'fit_{seed}_{method}.npz',p_by_task=bp,final_p=p,pred=pred,best_steps=bs)
   row=dict(seed=seed,method=method,parameters=340,steps=1200,gradient_calls=calls,best_steps=bs.tolist(),seconds=time.monotonic()-ts)
   for name,sl in [('train',slice(0,64)),('validation',slice(64,128)),('test',slice(128,None))]: row[name+'_mse']=((pred[:,sl]-y[:,sl])**2).mean(1).tolist()
   results.append(row); dump('results.json',results); dump('actual_finite_step.json',audits); print(json.dumps(row),flush=True)
 dump('receipt.json',dict(complete=len(results)==10,runs=len(results),seconds=time.monotonic()-start,pilot_sha256=hashlib.sha256(PILOT.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'protocol.md').read_bytes()).hexdigest()))
if __name__=='__main__': run()
