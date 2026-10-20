"""Bounded optimizer decision; historical source is read-only; no market access."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,time,json,hashlib
from pathlib import Path
import numpy as np
from autograd import grad,value_and_grad
from autograd.misc.flatten import flatten
from scipy.optimize import minimize
SOURCE=Path('E:/碳价格/独立数据归档/carbon-forecast-model-research/research/segmented_price_backbone_20261014')
sys.path.insert(0,str(SOURCE))
from model import initialize,predict,loss,train,step,D
from run_synthetic import generate
OUT=Path(__file__).parent
START=time.monotonic(); DEADLINE=START+900

def fit(x,y,vx,vy,seed,method):
 f,u=flatten(initialize(seed)); fun=lambda q:loss(u(q),x,y,'segmented'); fg=value_and_grad(fun)
 best=[float('inf'),f.copy(),0]; logs=[]; stopped='budget'; reason=''
 def record(q):
  value,g=fg(q); val=float(loss(u(q),vx,vy,'segmented')); i=len(logs)+1
  logs.append(dict(iteration=i,loss=float(value),validation=val,gradient_l2=float(np.linalg.norm(g)),gradient_inf=float(np.max(np.abs(g)))))
  if val<best[0]-1e-5:best[:]=[val,q.copy(),i]
  if time.monotonic()>DEADLINE:raise TimeoutError('global deadline')
  return i-best[2]>=20
 if method=='existing':
  # Unchanged original optimizer, fixed 60 epochs / 480 updates.
  p,h=train(x,y,'segmented',seed,DEADLINE); f,_=flatten(p);record(f); h['original_history']=h.pop('history'); stopped='fixed_budget_no_stationarity';reason='original 60 epochs / 480 updates'
 else:
  def callback(q):
   if record(q):raise StopIteration
  res=minimize(fg,f,jac=True,method='L-BFGS-B',callback=callback,options=dict(maxiter=200,maxfun=400,gtol=1e-4,ftol=1e-10,maxls=20))
  reason=str(res.message);stopped='validation_patience' if res.status==99 else ('gradient_stationarity' if np.max(np.abs(res.jac))<=1e-4 else 'function_tolerance' if res.success else 'bounded_limit_or_line_search')
  f=best[1];h=dict(updates=int(res.nit),evaluations=int(res.nfev))
 g=grad(fun)(f)
 return u(f),dict(method=method,stop=stopped,reason=reason,selected_gradient_l2=float(np.linalg.norm(g)),selected_gradient_inf=float(np.max(np.abs(g))),history=logs,**h)

def main():
 rows=[];scales=[];before=hashlib.sha256((SOURCE/'model.py').read_bytes()).hexdigest()
 for regime in ['nonlinear','null']:
  for seed in [7,17]:
   x,y,mean=generate(seed,regime);sx=max(np.std(x[:192]-x[:192,-1:]),.05);sy=np.maximum(np.std(y[:192],axis=0),.1);xx=x/sx;yy=y/sy
   p=initialize(seed); vv=(xx[:192]-xx[:192,-1:]).reshape(192,12,5); ee=np.maximum(vv@p['emb']+p['eb'],0);h=np.zeros((192,D)); gates=[]
   for t in range(12):
    a=ee[:,t]@p['wi']+p['bi'];b=h@p['wh']+p['bh'];gates.append(a[:,:2*D]+b[:,:2*D]);h=step(p,ee[:,t],h)
   scales.append(dict(regime=regime,seed=seed,input_scale=float(sx),target_scale=sy.tolist(),centered_input_std=float(np.std(vv)),embedding_std=float(np.std(ee)),gate_saturation_fraction=float(np.mean(np.abs(gates)>5))))
   for method in ['existing','lbfgsb']:
    t=time.monotonic();p,log=fit(xx[:192],yy[:192],xx[192:256],yy[192:256],seed,method);pred=predict(p,xx,'segmented')*sy
    row=dict(regime=regime,seed=seed,test_mse=float(np.mean((pred[256:]-y[256:])**2)),zero_mse=float(np.mean(y[256:]**2)),per_h_mse=np.mean((pred[256:]-y[256:])**2,0).tolist(),seconds=time.monotonic()-t,**log);rows.append(row)
    np.savez(OUT/f'{regime}_{seed}_{method}.npz',prediction=pred,target=y,input_scale=sx,target_scale=sy,**p)
    (OUT/'results.json').write_text(json.dumps(rows,indent=2));print(regime,seed,method,row['test_mse'],row['stop'],flush=True)
 assert len(rows)==8
 positive=all(next(r for r in rows if r['seed']==s and r['regime']=='nonlinear' and r['method']=='lbfgsb')['test_mse']<next(r for r in rows if r['seed']==s and r['regime']=='nonlinear' and r['method']=='existing')['test_mse'] for s in [7,17])
 null=all(r['test_mse']<=1.1*r['zero_mse'] for r in rows if r['regime']=='null' and r['method']=='lbfgsb')
 decision=dict(elapsed_seconds=time.monotonic()-START,positive_beats_existing_both=positive,null_gate_both=null,retain='autograd + scipy L-BFGS-B diagnostic base' if positive and null else 'abandon current handwritten GRU as trusted trainable base',market_refresh='not run; synthetic-only decision, no data access',source_sha256=before,source_unchanged=before==hashlib.sha256((SOURCE/'model.py').read_bytes()).hexdigest(),scales=scales,torch='not installed; mature CPU scipy optimizer used without handwritten optimization repair',split=[192,64,128],architecture_unchanged=True,innovation_claim=False)
 (OUT/'decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
