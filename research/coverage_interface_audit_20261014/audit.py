"""Finite interface audit: no original outputs modified, strict 950 prefix."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,importlib.util,hashlib,difflib
from pathlib import Path
import numpy as np
from scipy.linalg import solve
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
old=load('audit_state',ROOT/'research/multi_angle_model_search_20261013/state/run.py')
local=load('audit_local',ROOT/'research/local_coverage_model_20261014/run.py')
base=old.base;base.OUT=OUT
start=time.monotonic()
def dump(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
a=Path(old.__file__).read_text(encoding='utf-8');b=Path(local.__file__).read_text(encoding='utf-8')
(OUT/'SOURCE_DIFF.txt').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=str(old.__file__),tofile=str(local.__file__))),encoding='utf-8')
dates,p=base.read_prefix(); initial=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
if len(p)!=950 or len(orig)!=451:raise ValueError('support')
m,st=old.history(p,initial); truth=p[orig[:,None]+base.H]-p[orig,None];years=np.array([d[:4] for d in dates[orig]])
configs=[('historical_native','native','rms',True,10.),('historical_equation','none','rms',True,10.),('only_center','mean','rms',True,10.),('only_scale_std','none','std',True,10.),('only_free_intercept','none','rms',False,10.),('center_std_penalized','mean','std',True,10.),('local_fixed10','mean','std',False,10.),('local_fixed01','mean','std',False,.1),('local_selected','mean','std',False,'selected')]
preds={c[0]:np.zeros_like(truth) for c in configs};selections=[];logs=[]
def fit_predict(x,y,tr,te,center,scale,penalized,alpha):
 mu=x[tr].mean(0) if center=='mean' else np.zeros(x.shape[1]);sd=np.sqrt(np.mean(x[tr]**2,0)) if scale=='rms' else x[tr].std(0);sd[sd<1e-8]=1
 A=np.column_stack([np.ones(len(tr)),(x[tr]-mu)/sd]);T=np.column_stack([np.ones(len(te)),(x[te]-mu)/sd]);pen=np.eye(A.shape[1]);pen[0,0]=int(penalized)
 coef=solve(A.T@A+len(tr)*alpha*pen,A.T@y[tr],assume_a='pos');return T@coef
for end in range(initial,len(p),40):
 pos=np.where((orig>=end)&(orig<end+40))[0]
 if not len(pos):continue
 native,ll=old.correct(p,m,st,end,orig[pos],dict(family='linear',seed=None,alpha=10.));preds['historical_native'][pos]=native;logs.extend(ll)
 for j,h in enumerate(base.H):
  x=np.column_stack([m[:,j],st]);y=np.zeros(len(p));av=np.arange(len(p)-h);y[av]=p[av+h]-p[av]-m[av,j];idx=np.arange(max(120,end-int(h)-480),end-int(h))
  for name,center,scale,penalized,alpha in configs[1:]:
   chosen=alpha
   if alpha=='selected':
    cut=len(idx)-120;boundary=idx[cut];tr=idx[idx+h<boundary];va=idx[cut:];loss=[float(np.mean((fit_predict(x,y,tr,va,center,scale,penalized,v)-y[va])**2)) for v in [.1,10.]];chosen=[.1,10.][int(np.argmin(loss))];selections.append(dict(end=end,h=int(h),alpha=chosen,loss=loss))
   preds[name][pos,j]=m[orig[pos],j]+fit_predict(x,y,idx,orig[pos],center,scale,penalized,chosen)
  # Native local implementation independently checks the equation, with its knots machinery intact.
  for alpha,name in [(10.,'local_fixed10'),(.1,'local_fixed01')]:
   s=local.fit(x[idx],y[idx],'linear',alpha);native_local=m[orig[pos],j]+local.predict(s,x[orig[pos]])
   if np.max(np.abs(native_local-preds[name][pos,j]))>1e-9:raise ValueError('local equation mismatch')
 if time.monotonic()-start>550:raise TimeoutError('finite audit budget')
state_saved=np.load(Path(old.__file__).parent/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);local_saved=np.load(Path(local.__file__).parent/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
checks={'state_origins_equal':bool(np.array_equal(orig,state_saved['origins'])),'local_origins_equal':bool(np.array_equal(orig,local_saved['origins'])),'state_mle_maxdiff':float(np.max(np.abs(m-state_saved['issuance_mle']))),'local_mle_maxdiff':float(np.max(np.abs(m-local_saved['issuance_mle']))),'state_states_maxdiff':float(np.max(np.abs(st-state_saved['states']))),'local_states_maxdiff':float(np.max(np.abs(st-local_saved['states']))),'historical_saved_maxdiff':float(np.max(np.abs(preds['historical_native']-state_saved['linear_a10.0_sNone']))),'local10_saved_maxdiff':float(np.max(np.abs(preds['local_fixed10']-local_saved['linear_10.0']))),'local01_saved_maxdiff':float(np.max(np.abs(preds['local_fixed01']-local_saved['linear_0.1']))),'local_selected_saved_maxdiff':float(np.max(np.abs(preds['local_selected']-local_saved['linear_selected']))),'native_equation_maxdiff':float(np.max(np.abs(preds['historical_native']-preds['historical_equation'])))}
rows=[]
for name,pr in preds.items():
 score,by=base.score(pr,truth,years);rows.append(dict(name=name,score=score,by_year=by));print(name,score,flush=True)
if abs(rows[0]['score']-.9427054678157569)>1e-12:raise ValueError('historical reproduction failed')
if any(v>1e-9 for k,v in checks.items() if k.endswith('maxdiff')):raise ValueError('saved comparison failed')
np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',origins=orig,truth=truth,years=years,**preds)
dump('RESULTS.json',rows);dump('CHECKS.json',checks);dump('SELECTIONS.json',selections);dump('TRAINING_SUPPORT.json',logs);dump('RECEIPT.json',dict(seconds=time.monotonic()-start,rows=950,origins=451,initial=initial,read_boundary=False,source_sha256={str(Path(mod.__file__).relative_to(ROOT)):hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() for mod in [old,local,base]},configs=configs))
print(json.dumps(checks,indent=2))
