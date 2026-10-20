import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import importlib.util,json,csv,time,hashlib
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'joint_history'/'run.py'
spec=importlib.util.spec_from_file_location('parent_joint',SOURCE)
if spec is None or spec.loader is None: raise RuntimeError('source unavailable')
j=importlib.util.module_from_spec(spec); spec.loader.exec_module(j); b=j.b
METHODS=['independent_linear','joint_linear','group_sparse','scalar_validation_mix','horizon_shrinkage_baseline']
START=time.monotonic()
def fit(p,out=None):
 data,_=j.dataset(p); result={k:np.empty((len(data['test'][2]),3,3)) for k in METHODS}; receipt=[]
 for m in range(3):
  win={k:v[0].reshape(len(v[0]),31,3) for k,v in data.items()}
  xx={k:np.column_stack([v[:,:,m],v[:,:,[q for q in range(3) if q!=m]].reshape(len(v),62)]) for k,v in win.items()}
  mu=xx['train'].mean(0); sd=np.maximum(xx['train'].std(0),1e-8)
  x={k:np.column_stack([np.ones(len(v)),(v-mu)/sd]) for k,v in xx.items()}
  y={k:v[1][:,m*3:(m+1)*3] for k,v in data.items()}; n=len(y['train']); base=p[data['test'][2],m,None]
  chosen={}; hist={}; coefs={}
  for method,d in [('independent_linear',32),('joint_linear',94),('group_sparse',94)]:
   trials=[]
   for lam in ([.03,.3] if method=='group_sparse' else [10,100]):
    a=x['train'][:,:d]; gram=a.T@a/n; rhs=a.T@y['train']/n; pen=np.ones(d)*(10/n if method=='group_sparse' else lam/n); pen[0]=0
    if method=='group_sparse':
     pen[32:]=0; lips=np.linalg.eigvalsh(gram+np.diag(pen))[-1]; c=np.zeros((d,3)); converged=False
     for it in range(3000):
      if time.monotonic()-START>600: raise RuntimeError('fitting deadline')
      nxt=c-(gram@c-rhs+pen[:,None]*c)/lips
      for sl in [slice(32,63),slice(63,94)]:
       norm=np.linalg.norm(nxt[sl]); nxt[sl]*=max(0,1-lam/(lips*max(norm,1e-30)))
      change=float(np.linalg.norm(nxt-c)); c=nxt
      if change<1e-8*(1+np.linalg.norm(c)): converged=True; break
    else: c=np.linalg.solve(gram+np.diag(pen),rhs); it=0; converged=True; change=0.
    vp=x['val'][:,:d]@c; tp=x['test'][:,:d]@c; loss=float(np.mean((vp-y['val'])**2))
    trials.append((loss,vp,tp,c,{'parameter':lam,'validation_mse':loss,'iterations':it+1,'converged':converged,'iterate_change':change}))
   chosen[method]=min(trials,key=lambda z:z[0]); hist[method]=[z[4] for z in trials]; coefs[method]=chosen[method][3]
   result[method][:,m]=base+chosen[method][2]
  iv=chosen['independent_linear'][1]; itest=chosen['independent_linear'][2]; dv=chosen['joint_linear'][1]-iv; dt=chosen['joint_linear'][2]-itest; residual=y['val']-iv
  alpha=float(np.clip(np.sum(dv*residual)/max(np.sum(dv*dv),1e-30),0,1)); result['scalar_validation_mix'][:,m]=base+itest+alpha*dt
  lap=np.array([[1,-1,0],[-1,2,-1],[0,-1,1]]); Q=np.eye(3)+lap; energy=max(float(np.mean(dv*dv)),1e-12); trials=[]
  for frac in [.01,.1]:
   tau=frac*energy
   def fg(w):
    e=dv*w-residual
    return float(np.mean(e*e)+tau*w@Q@w),2*np.mean(dv*e,axis=0)/3+2*tau*Q@w
   opt=minimize(fg,np.full(3,alpha),jac=True,bounds=[(0,1)]*3,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':1000})
   w=opt.x; loss=float(np.mean((iv+dv*w-y['val'])**2)); trials.append((loss,w,{'fraction':frac,'validation_mse':loss,'success':bool(opt.success),'message':str(opt.message),'weights':w.tolist()}))
  selected=min(trials,key=lambda z:z[0]); result['horizon_shrinkage_baseline'][:,m]=base+itest+dt*selected[1]
  receipt.append({'market':m,'history':hist,'scalar_alpha':alpha,'horizon_trials':[z[2] for z in trials],'horizon_selected':selected[2]})
  if out:
   np.savez_compressed(out/f'market{m}_checkpoint.npz',mu=mu,sd=sd,**coefs,scalar_alpha=alpha,horizon_weights=selected[1])
   np.savez_compressed(out/f'market{m}_validation.npz',target=y['val'],independent=iv,joint=iv+dv,group_sparse=chosen['group_sparse'][1],origins=data['val'][2])
 return result,receipt

def main():
 rows=[]; aggregate={}; checks=[]; sources=[SOURCE,j.SOURCE,ROOT/'PROTOCOL.md',ROOT/'LITERATURE.md',ROOT/'run.py']; hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
 for seed in b.SEEDS:
  for shared in [True,False]:
   cond='shared' if shared else 'no_shared'; out=ROOT/f'seed_{seed}_{cond}'; out.mkdir(exist_ok=True); p,_=b.generate(seed,shared)
   if not np.array_equal(p,np.load(SOURCE.parent/out.name/'observations.npz')['prices']): raise RuntimeError('generator mismatch')
   pred,receipt=fit(p,out); ts=b.origins(len(p))['test']; target=np.stack([p[t+b.H] for t in ts]).transpose(0,2,1)
   (out/'training.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
   for method,v in pred.items():
    np.savez_compressed(out/f'{method}_predictions.npz',prediction=v,target=target,origins=ts); mse=np.mean((v-target)**2,axis=0)
    for m in range(3):
     for hi,h in enumerate(b.H): rows.append(dict(seed=seed,condition=cond,model=method,market=m,horizon=int(h),mse=float(mse[m,hi])))
   pert=p.copy(); pert[ts[0]+1:]+=10000; pred2,rec2=fit(pert)
   if receipt!=rec2 or any(not np.array_equal(v[0],pred2[k][0]) for k,v in pred.items()): raise RuntimeError('future leakage')
   checks.append({'seed':seed,'condition':cond,'all_refit_selections_and_first_origin_exactly_invariant':True})
   aggregate[f'{seed}_{cond}']={k:float(np.mean((v-target)**2)) for k,v in pred.items()}; print(seed,cond,aggregate[f'{seed}_{cond}'],flush=True)
 with (ROOT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
 saved=list(csv.DictReader((ROOT/'metrics.csv').open(encoding='utf-8')))
 if len(saved)!=180: raise RuntimeError('metric count')
 for r in saved:
  a=np.load(ROOT/f"seed_{r['seed']}_{r['condition']}"/f"{r['model']}_predictions.npz"); h=list(b.H).index(int(r['horizon'])); m=int(r['market']); score=float(np.mean((a['prediction'][:,m,h]-a['target'][:,m,h])**2))
  if not np.isclose(score,float(r['mse']),rtol=1e-12,atol=0): raise RuntimeError('reload mismatch')
 if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=s for p,s in hashes.items()): raise RuntimeError('source altered')
 harm={k:{m:v[m]-v['independent_linear'] for m in METHODS} for k,v in aggregate.items() if k.endswith('no_shared')}
 gate=all(aggregate[f'{s}_shared']['horizon_shrinkage_baseline']<min(aggregate[f'{s}_shared'][m] for m in ['scalar_validation_mix','group_sparse']) and harm[f'{s}_no_shared']['horizon_shrinkage_baseline']<=0 for s in b.SEEDS)
 verification={'rows':len(saved),'all_metrics_reload_verified':True,'source_hashes':hashes,'future_refits':checks,'by_seed':aggregate,'no_shared_excess_mse':harm,'empirical_gate':gate,'innovation_gate':False,'reason':'Exactly constrained generalized ridge on forecast corrections; ordinary shrinkage baseline only','wall_seconds':time.monotonic()-START,'train_last_label':int(b.origins(b.N)['train'][-1]+20),'val_first_origin':int(b.origins(b.N)['val'][0]),'val_last_label':int(b.origins(b.N)['val'][-1]+20),'test_first_origin':int(ts[0])}
 (ROOT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
 (ROOT/'RESULTS.md').write_text('# Conventional selective-transfer baseline results\n\nNOT RELEASED as innovation; exact generalized-ridge equivalence. No market access or threshold rescue.\n\n```json\n'+json.dumps(verification,indent=2)+'\n```\n\nAll no_shared market/horizon harm is reconstructible from the complete 180-row metrics.csv; aggregate excess shown above. Optimizer convergence for every sparse trial is in training.json; cap failure is not convergence. Same max-horizon maturity purge. Frozen validation-only weights, not online dynamic transfer.\n',encoding='utf-8')
 print(json.dumps(verification,indent=2))
if __name__=='__main__': main()
