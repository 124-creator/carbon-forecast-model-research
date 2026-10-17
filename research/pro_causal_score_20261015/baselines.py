"""Fresh standard baselines and mature-error combinations; no previous forecasts."""
import json,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import core

def run(path,out):
 out=Path(out);p,dates,raw=core.load_prefix(path)
 with np.load(out/'evaluation.npz',allow_pickle=False) as z:orig=z['origins'];truth=z['truth']
 def save(name,x):np.savez_compressed(out/'predictions'/f'{name}.npz',prediction=x[None] if x.ndim==2 else x)
 save('RW',np.zeros_like(truth));refs={}
 for w in [120,240,480]:refs[w]=core.reference(p,w);save(f'ML_{w}',refs[w]['b'][orig])
 for w in [120,240,480]:
  a=np.stack([(p[t]-p[max(0,t-w)])/(t-max(0,t-w))*np.arange(1,21) for t in orig]);save(f'DRIFT_{w}',a)
 register=[];start=time.perf_counter()
 configs=[('RIDGE',l,a) for l in [5,20] for a in [.01,.1,1,10]]+[('TREE',l,r) for l in [7,15] for r in [.05,.1]]
 for kind,a,b in configs:
  prediction=np.full_like(truth,np.nan)
  for block in range(12):
   end=459+40*block;indices=np.where((orig-459)//40==block)[0];ids,x,y,tr,va,sd=core.support(p,end)
   xx=x[:,-a:] if kind=='RIDGE' else x
   xnew=np.stack([np.diff(p[t-60:t+1]) for t in orig[indices]])
   if kind=='RIDGE':xnew=xnew[:,-a:]
   for h in [1,5,20]:
    reg=Ridge(alpha=b) if kind=='RIDGE' else HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=a,learning_rate=b,early_stopping=False,random_state=17)
    reg.fit(xx[tr]/sd,y[tr,h-1]/sd);prediction[indices,h-1]=reg.predict(xnew/sd)*sd
   register.append({'kind':kind,'a':a,'b':b,'block':block,'end':end,'train_max_target':int(ids[tr[-1]]+20),'horizons':[1,5,20]})
  save(f'{kind}_{a}_{b}',prediction)
 with np.load(out/'predictions/GRU_WINDOW.npz',allow_pickle=False) as z:slow=z['prediction']
 # combinations use the complete newly issued GRU archive, including unscored boundary origins
 issued=np.full((4,471,20),np.nan)
 for si,seed in enumerate(core.SEEDS):
  for block in range(12):
   with np.load(out/'fits'/f'GRU_WINDOW_s{seed}_b{block:02}.npz',allow_pickle=False) as z:issued[si,z['origins']-459]=z['prediction']
 base=refs[240]['b'];comb=(slow+base[orig][None])/2;save('CORE_GRU_FIXED',comb)
 weights=np.full((4,451,20),.5)
 for si in range(4):
  for oi,t in enumerate(orig):
   for h in range(1,21):
    past=np.arange(max(459,t-h-120),t-h)
    if len(past)<30:continue
    yy=p[past+h]-p[past];bhat=base[past,h-1];delta=issued[si,past-459,h-1]-bhat
    denom=float(delta@delta)
    if denom>1e-12:weights[si,oi,h-1]=np.clip(delta@(yy-bhat)/denom,0,1)
 pred=base[orig][None]+weights*(slow-base[orig][None]);save('CORE_GRU_MATURE',pred)
 np.savez_compressed(out/'combination_weights.npz',weights=weights)
 (out/'BASELINE_REGISTER.json').write_text(json.dumps({'fits':register,'configuration_count':19,'plus_combinations':2,'wall_seconds':time.perf_counter()-start,'placeholder_horizons':'NaN, not forecasted'},indent=2))
