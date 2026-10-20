import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import spline
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'research/deep_direction_review_20261013/calibration'));import calibrate as base
import importlib.util
spec=importlib.util.spec_from_file_location('state_baseline',ROOT/'research/multi_angle_model_search_20261013/state/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);base.OUT=OUT
base.OUT=OUT;START=time.monotonic();P=json.loads((OUT/'PROTOCOL.json').read_text());LOG=[];SAVED={}
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def check(c,m):
 if not c:raise ValueError(m)
def select(x,y,kind,label):
 check(time.monotonic()-START<900,'900 second budget')
 cut=len(x)-120;train_stop=cut-20;hs=np.maximum(np.sqrt(np.mean(y[:train_stop]**2,axis=0)),1e-8);rows=[]
 for a in P['alpha_grid']:
  s=spline.fit(x[:train_stop],y[:train_stop]/hs,kind,a);e=float(np.mean(((spline.predict(s,x[cut:])*hs-y[cut:])/hs)**2));rows.append(dict(alpha=a,validation_mse=e,gradient_norm=s['gradient_norm']))
 alpha=min(rows,key=lambda q:q['validation_mse'])['alpha'];hs=np.maximum(np.sqrt(np.mean(y*y,axis=0)),1e-8)
 s=spline.fit(x,y/hs,kind,alpha);LOG.append(dict(label=label,kind=kind,all_configurations=rows,selected_alpha=alpha,parameters=s['parameters'],effective_df=s['effective_df'],gradient_norm=s['gradient_norm'],training_mse=s['training_mse']))
 return s,hs
def blocks(p):
 raw=base.raw_features(p);return np.column_stack([raw[:,:1].mean(1),raw[:,1:5].mean(1),raw[:,5:20].mean(1),raw[:,20:].mean(1)])
def synthetic():
 rows=[]
 for seed in P['seeds']:
  r=np.random.default_rng(seed);raw=r.normal(size=(2480,60));x=np.column_stack([raw[:,:1].mean(1),raw[:,1:5].mean(1),raw[:,5:20].mean(1),raw[:,20:].mean(1)]);z=x*np.sqrt([1,4,15,40]);noise=r.normal(size=(2480,3))*.15
  for task in ['additive','interaction','null']:
   f=(z[:,0]**2-1)+.5*np.maximum(z[:,2],0) if task=='additive' else 1.5*z[:,0]*z[:,1] if task=='interaction' else np.zeros(2480)
   y=f[:,None]*np.array([1,2,3])+noise;rw=float(np.mean(y[480:]**2));res={}
   SAVED[f'synthetic_{task}_{seed}_truth']=y[480:]
   for kind in P['kinds']:
    s,hs=select(x[:480],y[:480],kind,f'synthetic_{task}_{seed}');pr=spline.predict(s,x[480:])*hs;mse=float(np.mean((pr-y[480:])**2));res[kind]=mse/rw;SAVED[f'synthetic_{task}_{seed}_{kind}']=pr
   rows.append(dict(seed=seed,task=task,ratio_rw=res))
 dump('SYNTHETIC.json',rows)
 passed=all((v['ratio_rw']['additive']<.6 and v['ratio_rw']['additive']/v['ratio_rw']['linear']<.8) if v['task']=='additive' else (v['ratio_rw']['interaction']/v['ratio_rw']['additive']<.8) if v['task']=='interaction' else max(v['ratio_rw'].values())<=1.10 for v in rows)
 dump('GATE.json',dict(passed=passed,market_allowed=passed));return passed

def market():
 dates,p=base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));origins=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]]);check(len(origins)==451,'451 origins')
 truth=p[origins[:,None]+base.H]-p[origins,None];years=np.array([v[:4] for v in dates[origins]]);x=blocks(p);issuance,states=old.history(p,initial);pred={k:np.full_like(truth,np.nan) for k in P['kinds']+['RW','MLE_window480','linear_correction']};leak=[]
 SAVED.update(origins=origins,truth=truth,years=years,origin_prices=p[origins])
 for end in range(initial,len(p),40):
  pos=np.where((origins>=end)&(origins<end+40))[0]
  if not len(pos):continue
  idx=np.arange(max(60,end-20-480),end-20);check(idx[-1]+20<end,'maturity');y=p[idx[:,None]+base.H]-p[idx,None]
  for kind in P['kinds']:
   s,hs=select(x[idx],y,kind,f'market_{end}');pr=spline.predict(s,x[origins[pos]])*hs;pred[kind][pos]=pr
   for key,value in s.items():
    if isinstance(value,np.ndarray):SAVED[f'{kind}_{end}_{key}']=value
   SAVED[f'{kind}_{end}_hs']=hs
   q=p.copy();q[end:]+=np.arange(len(p)-end)*1000+111
   # Perturb all unobserved prices after fit origin end-1. Compare forecast at end-1.
   xq=blocks(q);sq,hq=select(xq[idx],q[idx[:,None]+base.H]-q[idx,None],kind,f'perturb_{end}')
   e=float(np.max(np.abs(spline.predict(s,x[end-1:end])*hs-spline.predict(sq,xq[end-1:end])*hq)))
   leak.append(dict(end=end,kind=kind,max_difference=e,max_training_target=int(idx[-1]+20)))
  pred['RW'][pos]=0;pred['MLE_window480'][pos]=base.predict(p,end,origins[pos],dict(family='MLE',window=480))[0];pred['linear_correction'][pos]=old.correct(p,issuance,states,end,origins[pos],dict(family='linear',alpha=10.,seed=None))[0]
 rows=[]
 for kind,pr in pred.items():
  check(np.isfinite(pr).all(),'finite');score,by=base.score(pr,truth,years);rows.append(dict(name=kind,score=score,by_year=by));SAVED[kind]=pr
 check(max(v['max_difference'] for v in leak)<1e-10,'future perturbation');dump('CAUSALITY.json',leak);dump('RESULTS.json',dict(results=rows,rows=950,origins=451,development_only=True,seconds=time.monotonic()-START));return rows
if __name__=='__main__':
 ok=synthetic();rows=market() if ok else []
 dump('TRAINING_LOGS.json',LOG);np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**SAVED)
 dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-START,protocol_sha256=hashlib.sha256((OUT/'PROTOCOL.json').read_bytes()).hexdigest(),gate_passed=ok,market_run=bool(rows),numpy=np.__version__))
 print(json.dumps(dict(gate=ok,results=rows,seconds=time.monotonic()-START),ensure_ascii=False,indent=2))
