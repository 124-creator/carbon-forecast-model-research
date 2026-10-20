import importlib.util,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('run',R/'run.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
rows=json.loads((R/'results.json').read_text()); errors=[]; metric_count=0
for row in rows:
 seed=row['seed']; method=row['method']; fit=np.load(R/f'fit_{seed}_{method}.npz'); x,y=m.pilot.data(seed); stored=np.load(R/f'data_{seed}.npz')
 if not np.array_equal(x,stored['x']) or not np.array_equal(y,stored['y']): errors.append('data')
 arch='independent' if method=='independent' else 'shared_private'
 pred=np.array([m.pilot.forward(fit['p_by_task'][t],x,arch)[0][t] for t in range(4)])
 if np.max(abs(pred-fit['pred']))>1e-12: errors.append('prediction')
 for name,sl in [('train',slice(0,64)),('val',slice(64,128)),('test',slice(128,None))]:
  val=((pred[:,sl]-y[:,sl])**2).mean(1); metric_count+=4
  if np.max(abs(val-row[name+'_mse']))>1e-12: errors.append('metric')
# Private isolation: perturb other task labels, target private slices cannot change.
x,y=m.pilot.data(17); p=m.pilot.init(17); g=m.pilot.objective(p,x[:,:64],y[:,:64],'shared_private',0)[1]; yy=y[:,:64].copy(); yy[1:]+=100; h=m.pilot.objective(p,x[:,:64],yy,'shared_private',0)[1]
if not np.array_equal(g[~m.mask],h[~m.mask]): errors.append('private isolation')
# Projection witness: exact pseudo-task QP/KKT; zero rows, opposite gradients.
G=np.array([[1.,0.],[-1.,0.],[0.,1.]]); d,feas,kkt=m.project(G,np.array([1.,2.]))
if np.max(abs(d-np.array([0.,2.])))>1e-8: errors.append('QP witness')
a=json.loads((R/'actual_block_losses.json').read_text()); out=dict(runs=len(rows),metrics_checked=metric_count,errors=errors,actual_loss_steps=len(a),min_constraint_dot=min(z['min_normalized_dot'] for z in a),max_kkt=max(z['kkt_complementarity'] for z in a),shared_only_risk_increase_steps=sum(np.max(np.array(z['after_shared_only'])-z['before'])>1e-10 for z in a),joint_risk_increase_steps=sum(np.max(np.array(z['after_joint'])-z['before'])>1e-10 for z in a),private_isolation=True,files=[dict(name=f.name,bytes=f.stat().st_size) for f in R.iterdir() if f.is_file()])
(R/'verification.json').write_text(json.dumps(out,indent=2,default=lambda v: v.item())); print(json.dumps(out,ensure_ascii=False,default=lambda v: v.item()));
if errors: raise RuntimeError(errors)
