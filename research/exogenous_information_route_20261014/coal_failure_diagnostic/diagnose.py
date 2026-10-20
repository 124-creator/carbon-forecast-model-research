import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,importlib.util,csv,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent; SRC=OUT.parent/'coal_training'
spec=importlib.util.spec_from_file_location('coal_diag_source',SRC/'run.py')
if spec is None or spec.loader is None:raise RuntimeError('source module unavailable')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.base.OUT=OUT

def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def corr(a,b):
 a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den>1e-12 else None

def main():
 begin=time.monotonic();z=np.load(SRC/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);logs=json.loads((SRC/'TRAINING_LOGS.json').read_text(encoding='utf8'))
 dates,p=m.base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));B=m.vintage(p,initial);orig=z['origins'];H=m.H
 if not np.array_equal(B[orig],z['MLE']):raise ValueError('vintage replay')
 Y=np.zeros_like(B);ii=np.arange(len(p)-20);Y[ii]=p[ii[:,None]+H]-p[ii,None];R=Y-B
 raw=m.base.raw_features(p);widths=[1,1,1,2,5,10,20,20];Q=np.zeros((8,60));start=0
 for j,w in enumerate(widths):Q[j,start:start+w]=1/np.sqrt(w);start+=w
 P=raw@Q.T;blocks=[];decomp=[];directions=[];parts={}
 for lag in (1,2):
  X=z['features_lag'+str(lag)];valid=z['valid_lag'+str(lag)];age=z['age_sessions_lag'+str(lag)]
  for name in ('linear','kernel'):
   key=f'{name}_lag{lag}';D=m.design(X,age,name);comp=np.zeros((len(orig),3,5));replay=z['MLE'].copy()
   for l in [l for l in logs if l['lag']==lag and l['name']==name]:
    end=l['end'];idx=np.arange(max(60,end-20-480),end-20);idx=idx[valid[idx]];boundary=idx[-80];tr=idx[idx+20<boundary];va=idx[-80:];a=l['alpha'];s=m.fit(D,R,idx,a,name);mu,sd,ym,b=s
    pos=np.where((orig>=end)&(orig<end+40)&valid[orig])[0];dd=D[orig[pos]]
    if name=='linear':
     contrib=((dd[:,0]-mu)/sd)[:,:,None]*b[None,:,:];contrib=contrib.transpose(0,2,1);comp[pos,:,0]=ym
    else:
     contrib=((dd-mu)/sd)*b;comp[pos,:,0]=ym
    comp[pos,:,1:]=contrib;replay[pos]+=comp[pos].sum(2)
    A=D[idx,0] if name=='linear' else D[idx].reshape(-1,4);ZZ=(A-mu)/sd;sv=np.linalg.svd(ZZ,compute_uv=False);rank=int(np.linalg.matrix_rank(ZZ));nonzero=sv[sv>1e-10]
    blocks.append(dict(model=key,end=end,date=str(dates[end]),seed=None,n=len(idx),inner_n=len(tr),validation_n=len(va),alpha=a,mean=mu.tolist(),scale=sd.tolist(),rank=rank,condition_active=float(nonzero[0]/nonzero[-1]),condition_penalized=float(np.linalg.cond(ZZ.T@ZZ+len(A)*a*np.eye(4))),intercept=np.asarray(ym).tolist(),beta=np.asarray(b).tolist(),unique_feature_rows=int(len(np.unique(X[idx,:3],axis=0))),n_training_event_changes=int(np.sum(np.any(np.diff(X[idx,:2],axis=0)!=0,axis=1))),shock_mean=float(X[idx,1].mean()),shock_std=float(X[idx,1].std())))
    # Fixed alpha inherited from the original log; no alpha search and no production changes.
    # Price nuisance and coal partialling use only the original purged train segment.
    pm=P[tr].mean(0);ps=np.maximum(P[tr].std(0),1e-10);pt=(P[tr]-pm)/ps;pv=(P[va]-pm)/ps
    rm=R[tr].mean(0);rb=np.linalg.solve(pt.T@pt+len(tr)*a*np.eye(8),pt.T@(R[tr]-rm));rt=R[tr]-(pt@rb+rm);rv=R[va]-(pv@rb+rm)
    xm=X[tr,:3].mean(0);xs=np.maximum(X[tr,:3].std(0),1e-10);xt=(X[tr,:3]-xm)/xs;xv=(X[va,:3]-xm)/xs
    xb=np.linalg.solve(pt.T@pt+len(tr)*a*np.eye(8),pt.T@xt);ut=xt-pt@xb;uv=xv-pv@xb
    cb=np.linalg.solve(ut.T@ut+len(tr)*a*np.eye(3),ut.T@rt);ct=ut@cb;cv=uv@cb
    inner=m.fit(D,R,tr,a,name);ic=m.predict(inner,D[va],name)
    for j,h in enumerate(H):
     directions.append(dict(model=key,end=end,h=int(h),seed=None,train_residual_mean=float(R[tr,j].mean()),validation_residual_mean=float(R[va,j].mean()),coal_raw_train_corr=[corr(X[tr,k],R[tr,j]) for k in range(3)],coal_raw_validation_corr=[corr(X[va,k],R[va,j]) for k in range(3)],orth_train_corr=[corr(ut[:,k],rt[:,j]) for k in range(3)],orth_validation_corr=[corr(uv[:,k],rv[:,j]) for k in range(3)],orth_train_alignment=float(np.mean(ct[:,j]*rt[:,j])),orth_validation_alignment=float(np.mean(cv[:,j]*rv[:,j])),orth_validation_delta=float(np.mean(cv[:,j]**2-2*cv[:,j]*rv[:,j])),original_inner_delta=float(np.mean(ic[:,j]**2-2*ic[:,j]*R[va,j])),priceonly_inner_delta=float(np.mean((pv@rb+rm)[:,j]**2-2*(pv@rb+rm)[:,j]*R[va,j]))))
   delta=float(np.max(np.abs(replay-z[key])))
   if delta>1e-10:raise ValueError('original correction replay '+key)
   parts[key]=comp
   for year in ('2020','2021'):
    mask=z['years']==year;err=z['truth'][mask]-z['MLE'][mask];c=comp[mask].sum(2);den=np.mean(z['truth'][mask]**2,0)
    for j,h in enumerate(H):
     cross=np.mean(comp[mask,j,:]*c[:,j,None],axis=0);align=np.mean(comp[mask,j,:]*err[:,j,None],axis=0);change=float(np.mean(c[:,j]**2-2*c[:,j]*err[:,j]))
     decomp.append(dict(model=key,year=year,h=int(h),seed=None,n=int(mask.sum()),valid_n=int(valid[orig[mask]].sum()),correction_mean=float(c[:,j].mean()),error_mean=float(err[:,j].mean()),correction_rms=float(np.sqrt(np.mean(c[:,j]**2))),alignment=float(np.mean(c[:,j]*err[:,j])),energy=float(np.mean(c[:,j]**2)),mse_change=change,relative_change=change/float(den[j]),component_names=['intercept','level','shock_difference','age','missing'],component_mean=comp[mask,j,:].mean(0).tolist(),component_alignment=align.tolist(),component_additive_mse_change=(cross-2*align).tolist(),identity_error=abs(float((cross-2*align).sum())-change),replay_max=delta))
 dump('BLOCK_SUPPORT_SCALES.json',blocks);dump('YEAR_H_DECOMPOSITION.json',decomp);dump('PURGED_DIRECTION_DIAGNOSTICS.json',directions);np.savez_compressed(OUT/'COMPONENTS_LOCAL_ONLY.npz',origins=orig,**parts)
 summary=dict(blocks=len(blocks),cells=len(decomp),direction_cells=len(directions),max_replay=max(x['replay_max'] for x in decomp),max_identity_error=max(x['identity_error'] for x in decomp),alpha_counts={str(a):sum(b['alpha']==a for b in blocks) for a in (.1,10)},seed=None,price_last=str(dates[-1]),no_new_hyperparameter_selection=True,seconds=time.monotonic()-begin)
 dump('VERIFICATION_OPT.json' if not __debug__ else 'VERIFICATION_NORMAL.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
