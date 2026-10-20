"""Synthetic-only fixed random encoder estimation comparison; 600s global cap."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnostic import Encoder,data,ridge,L,S,D
start=time.monotonic(); root=Path(__file__).resolve().parent
train,y=data(203,300); test,yt=data(204,200)
np.savez(root/'synthetic_inputs.npz',train=train,y=y,test=test,yt=yt)
rows=[]; preds={}; weights={}
for seed in [11,22,33]:
 e=Encoder(seed); ht=e.bank(train[:,2]); hs=e.bank(test.reshape(-1,L)).reshape(200,S,S,D)
 mean=ht.mean(1); dev=ht-mean[:,None,:]
 # Mean-loss plus conditional phase residual variance is an estimator, not pooling.
 for lam in [0.,.01,.1,1.]:
  a=np.column_stack([mean,np.ones(len(y))]); reg=np.eye(D+1)*.01; reg[-1,-1]=0
  cov=np.zeros((D+1,D+1)); cov[:D,:D]=np.einsum('npd,npe->de',dev,dev)/S
  w=np.linalg.solve(a.T@a+reg+lam*cov,a.T@y)
  for infer in (['mean'] if lam==0 else ['mean','hard']):
   f=hs.mean(2) if infer=='mean' else hs[:,:,0]
   name='average' if lam==0 else f'conditional_consistency_{lam}_{infer}'
   pred=f@w[:-1]+w[-1]; mse=((pred-yt[:,None])**2).mean(0)
   rows.append(dict(seed=seed,model=name,worst_mse=float(mse.max()),mean_mse=float(mse.mean()),per_phase_mse=mse.tolist(),phase_sd=float(pred.std(1).mean()),head_parameters=17))
   preds[f'{seed}_{name}']=pred; weights[f'{seed}_{name}']=w
 for name,f in [('hard',e.hard),('sliding',e.slide)]:
  w=ridge(f(train[:,2]),y); pred=f(test.reshape(-1,L)).reshape(200,S,D)@w[:-1]+w[-1]
  mse=((pred-yt[:,None])**2).mean(0); rows.append(dict(seed=seed,model=name,worst_mse=float(mse.max()),mean_mse=float(mse.mean()),per_phase_mse=mse.tolist(),phase_sd=float(pred.std(1).mean()),head_parameters=17)); preds[f'{seed}_{name}']=pred; weights[f'{seed}_{name}']=w
 # Learned weighting tied to readout w, no extra free parameters; analytic gradient.
 from scipy.optimize import minimize
 def loss(w):
  z=ht@w[:-1]; q=np.exp(z-z.max(1,keepdims=True)); q/=q.sum(1,keepdims=True)
  mu=(q*z).sum(1); err=mu+w[-1]-y
  grad=(q[...,None]*(1+z-mu[:,None])[...,None]*ht).sum(1)
  return float((err**2).mean()+.01/300*(w[:-1]**2).sum()),np.r_[2*(err[:,None]*grad).mean(0)+2*.01/300*w[:-1],2*err.mean()]
 opt=minimize(loss,ridge(mean,y),jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-10})
 w=opt.x; z=hs@w[:-1]; q=np.exp(z-z.max(2,keepdims=True)); q/=q.sum(2,keepdims=True); pred=(q*z).sum(2)+w[-1]; mse=((pred-yt[:,None])**2).mean(0)
 name='tied_learned_weight'; rows.append(dict(seed=seed,model=name,worst_mse=float(mse.max()),mean_mse=float(mse.mean()),per_phase_mse=mse.tolist(),phase_sd=float(pred.std(1).mean()),head_parameters=17,optimizer_success=bool(opt.success),iterations=int(opt.nit))); preds[f'{seed}_{name}']=pred; weights[f'{seed}_{name}']=w
 if time.monotonic()-start>600: raise RuntimeError('global 600-second cap exceeded')
np.savez(root/'predictions.npz',**preds); np.savez(root/'heads.npz',**weights)
aggregate={n:float(np.mean([r['worst_mse'] for r in rows if r['model']==n])) for n in sorted(set(r['model'] for r in rows))}
out=dict(rows=rows,aggregate_worst_mse=aggregate,elapsed_seconds=time.monotonic()-start,budget_seconds=600,scope='Frozen random encoder; estimator training only, not PMF/end-to-end. Circular phase bank inherited for exact prior-task comparison. No market reads.',seeds=[11,22,33],selection='lambda grid diagnostics, no test-based selection; lambda .1 predeclared candidate',free_parameters=17,total_parameters=1793)
(root/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out['aggregate_worst_mse'],indent=2)); print('elapsed',out['elapsed_seconds'])
