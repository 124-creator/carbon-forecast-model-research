import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import importlib.util,json,time,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
src=R.parent/'pilot/run_pilot.py'
s=importlib.util.spec_from_file_location('pilot',src); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
mask=np.concatenate([np.tile(np.arange(12)<6,20),np.tile(np.arange(12)<6,4),np.tile(np.arange(12)<6,4),np.zeros(4,dtype=bool)])
methods=['SP','PCGrad','adaptive_weights','temporal_transfer','random_tail','private_only']
def dump(n,v): (R/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
def gradients(p,x,y): return np.array([m.objective(p,x,y,'shared_private',i)[1] for i in range(4)])
def update(p,G,H,method,rng):
 A=H[:,mask]@G[:,mask].T
 score=A.mean(0)/max(np.sqrt(np.mean(A*A)),1e-12); w=np.exp(score-score.max()); w/=w.sum()
 g=G.mean(0)
 if method=='PCGrad':
  Q=G[:,mask].copy(); orig=Q.copy()
  for i in range(4):
   for j in rng.permutation(4):
    d=Q[i]@orig[j]
    if d<0: Q[i]-=d/(orig[j]@orig[j]+1e-30)*orig[j]
  g[mask]=Q.mean(0)
 elif method=='adaptive_weights': g=w@G
 elif method in ['temporal_transfer','random_tail']: g[mask]=w@G[:,mask]
 elif method=='private_only': g[mask]=0
 return p-.04*g,A,w

def run():
 start=time.monotonic(); rows=[]; protocol=hashlib.sha256((R/'protocol.md').read_bytes()).hexdigest()
 for seed in [17,29]:
  x,y=m.data(seed); x=(x-x[:,:48].mean(1,keepdims=True))/x[:,:48].std(1,keepdims=True)
  np.savez_compressed(R/f'data_{seed}.npz',x=x,y=y)
  ri=np.random.default_rng(seed+555).choice(48,16,replace=False)
  for method in methods:
   p=m.init(seed); bp=np.tile(p,(4,1)); best=np.full(4,np.inf); bs=np.zeros(4,int); trace=[]; rng=np.random.default_rng(seed+333); ts=time.monotonic()
   for step in range(1201):
    if time.monotonic()-start>600: raise RuntimeError('total training cap 600sec')
    if step%20==0:
     val=((m.forward(p,x[:,64:128],'shared_private')[0]-y[:,64:128])**2).mean(1); ok=val<best; best[ok]=val[ok]; bp[ok]=p; bs[ok]=step
    if step==1200: break
    G=gradients(p,x[:,:48],y[:,:48]); ids=ri if method=='random_tail' else np.arange(48,64)
    H=gradients(p,x[:,ids],y[:,ids]); p,A,w=update(p,G,H,method,rng)
    if step%20==0: trace.append(dict(step=step,tail_transfer=A.tolist(),weights=w.tolist(),shared_norm=float(np.linalg.norm(p[mask])),core_conflict_fraction=float(np.mean((G[:,mask]@G[:,mask].T)<0))))
   pred=np.array([m.forward(bp[t],x,'shared_private')[0][t] for t in range(4)])
   np.savez_compressed(R/f'fit_{seed}_{method}.npz',p_by_task=bp,final_p=p,pred=pred,best_steps=bs)
   row=dict(seed=seed,method=method,steps=1200,gradient_calls=9600,seconds=time.monotonic()-ts,best_steps=bs.tolist())
   for n,sl in [('core',slice(0,48)),('inner_tail',slice(48,64)),('outer_validation',slice(64,128)),('test',slice(128,None))]: row[n+'_mse']=((pred[:,sl]-y[:,sl])**2).mean(1).tolist()
   rows.append(row); dump(f'trace_{seed}_{method}.json',trace); dump('results.json',rows); print(json.dumps(row),flush=True)
 dump('receipt.json',dict(runs=len(rows),complete=len(rows)==12,seconds=time.monotonic()-start,protocol_sha256=protocol,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pilot_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),market_access=False))
if __name__=='__main__': run()
