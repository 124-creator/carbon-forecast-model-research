import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import time,json
from pathlib import Path
from model import initialize,predict,loss,train
from autograd import grad
from autograd.misc.flatten import flatten
OUT=Path(__file__).parent
def generate(seed,regime):
 r=np.random.RandomState(seed+2000); z=r.uniform(-1,1,384)
 x=np.arange(60)[None,:]*z[:,None]/60+r.normal(0,.015,(384,60))
 mean=np.abs(z)[:,None]*np.array([.2,.5,1.])[None,:]
 if regime=='null': mean=np.zeros((384,3))
 y=mean+r.normal(0,.1,(384,3))
 return x,y,mean
def main():
 if (OUT/'RESULTS.json').exists(): raise RuntimeError('preserve first run')
 start=time.monotonic(); deadline=start+900; rows=[]; checks=[]
 for regime in ['nonlinear','null']:
  for seed in [7,17]:
   x,y,mean=generate(seed,regime); sx=max(np.std(x[:256]-x[:256,-1:]),.05); sy=np.maximum(np.std(y[:256],axis=0),.1)
   xx=x/sx; yy=y/sy; X=np.column_stack([np.ones(384),xx-xx[:,-1:]])
   pen=np.eye(61);pen[0,0]=0
   coef=np.linalg.solve(X[:256].T@X[:256]+pen,X[:256].T@yy[:256])
   for mode in ['linear','segmented','unsegmented']:
    t=time.monotonic()
    if mode=='linear': pred=X@coef*sy; log={}
    else:
     p=initialize(seed); f,u=flatten(p); fun=lambda ff:loss(u(ff),xx[:8],yy[:8],mode)
     dg=grad(fun)(f); direction=np.random.RandomState(23).normal(size=len(f));direction/=np.linalg.norm(direction)
     fd=(fun(f+1e-5*direction)-fun(f-1e-5*direction))/2e-5;err=float(abs(fd-dg@direction))
     if err>1e-5: raise RuntimeError('gradient failed')
     checks.append(dict(regime=regime,seed=seed,model=mode,direction_gradient_error=err))
     p,log=train(xx[:256],yy[:256],mode,seed,deadline)
     pred=predict(p,xx,mode)*sy
    row=dict(regime=regime,seed=seed,model=mode,train_mse=float(np.mean((pred[:256]-y[:256])**2)),mse=float(np.mean((pred[256:]-y[256:])**2)),zero_mse=float(np.mean(y[256:]**2)),per_h_mse=np.mean((pred[256:]-y[256:])**2,0).tolist(),seconds=time.monotonic()-t,**log)
    rows.append(row);np.savez(OUT/f'{regime}_{seed}_{mode}.npz',prediction=pred,target=y,mean=mean,input_scale=sx,target_scale=sy)
    (OUT/'RESULTS.json').write_text(json.dumps(rows,indent=2)); print(regime,seed,mode,row['mse'],row['seconds'],flush=True)
 gates={}
 for seed in [7,17]:
  rr={r['model']:r for r in rows if r['regime']=='nonlinear' and r['seed']==seed};a=rr['segmented']
  gates[f'positive_{seed}']=bool(a['mse']<.9*rr['linear']['mse'] and a['mse']<.7*a['zero_mse'] and a['history'][-1]<.5*a['initial'])
  a=next(r for r in rows if r['regime']=='null' and r['seed']==seed and r['model']=='segmented')
  gates[f'null_{seed}']=bool(a['mse']<=1.1*a['zero_mse'])
 gates['synthetic_pass']=all(gates.values());gates['elapsed_seconds']=time.monotonic()-start
 (OUT/'GATES.json').write_text(json.dumps(gates,indent=2));(OUT/'CHECKS.json').write_text(json.dumps(checks,indent=2));print(gates)
if __name__=='__main__':main()
