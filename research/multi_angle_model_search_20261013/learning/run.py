"""Fixed-budget diagnostic; polynomial ridge with ESS-selected penalty, not innovation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[1]/'deep_direction_review_20261013/calibration'))
import calibrate as base
base.OUT=OUT
H=base.H

def ess(e):
 e=e-e.mean();den=e@e
 if den<1e-14:return float(len(e))
 rho=[max(0.,float(e[:-k]@e[k:]/den)) for k in range(1,min(20,len(e)-1)+1)]
 return float(len(e)/(1+2*sum(rho)))

def ridge(A,Y,cap=None):
 n,d=A.shape;ev=np.maximum(np.linalg.eigvalsh(A.T@A),0);lam=float(n)
 df=lambda l:float(np.sum(ev/(ev+l)))
 if cap is not None and df(lam)>cap:
  lo=lam;hi=lam
  while df(hi)>cap:hi*=2
  for _ in range(60):
   mid=(lo+hi)/2
   if df(mid)>cap:lo=mid
   else:hi=mid
  lam=hi
 B=np.linalg.solve(A.T@A+lam*np.eye(d),A.T@Y)
 return B,lam,df(lam)

def prepare(p):
 n=len(p);bank=np.zeros((n,31,3));ll=np.zeros((n,31,2));X=np.zeros((n,8));vintage=np.zeros((n,3))
 widths=[1,1,1,2,5,10,20,19]
 for t in range(59,n):
  b,a,_=base.filter_bank(p[t-59:t+1]);bank[t]=b[-1]
  if t+1<n:ll[t]=base.filter_bank(p[t-59:t+2])[1][-1]
  inc=np.diff(p[t-59:t+1])[::-1];s=0
  for j,w in enumerate(widths):X[t,j]=inc[s:s+w].sum()/np.sqrt(w);s+=w
 # Issuance-vintage parameters: refresh anchored at earliest feasible mature fit.
 for fit in range(120,n,40):
  idx=np.arange(max(60,fit-20-480),fit-20);a=ll[idx].sum(0)
  j=int(np.argmin(a[:,0]+len(idx)*np.log(np.maximum(a[:,1]/len(idx),1e-14))))
  vintage[fit:min(fit+40,n)]=bank[fit:min(fit+40,n),j]
 return X,bank,ll,vintage

def block(p,fit,orig,prepared):
 X,bank,ll,vintage=prepared;idx=np.arange(max(120,fit-20-480),fit-20)
 if idx[-1]+20>=fit:raise ValueError('maturity')
 a=ll[idx].sum(0);j=int(np.argmin(a[:,0]+len(idx)*np.log(np.maximum(a[:,1]/len(idx),1e-14))))
 baseline=bank[orig,j];Y=p[idx[:,None]+H]-p[idx,None]-vintage[idx]
 mu=X[idx].mean(0);sd=X[idx].std(0);sd[sd<1e-12]=1
 Z=(X-mu)/sd;i,jj=np.triu_indices(8);F=np.column_stack([Z,Z[:,i]*Z[:,jj]])
 ens=[ess(Y[:,h]) for h in range(3)];cap=min(44.,min(ens)/10)
 out={'MLE_matched':baseline};logs=[]
 for name,d,c in [('residual_ridge',8,None),('fixed_interaction',44,None),('ESS_interaction',44,cap)]:
  A=F[idx,:d];m=A.mean(0);s=A.std(0);s[s<1e-12]=1;A=(A-m)/s;T=(F[orig,:d]-m)/s
  # Zero correction mean: no unpenalized intercept escaping ESS cap.
  B,lam,df=ridge(A,Y,c);out[name]=baseline+T@B
  logs.append(dict(name=name,fit=fit,min_origin=int(idx[0]),max_origin=int(idx[-1]),max_label=int(idx[-1]+20),n=len(idx),ess_h=ens,cap=c,lambda_sum_loss=lam,df=df,coefficients=B.tolist(),selected_mle_grid=int(j)))
 strong,_=base.predict(p,fit,orig,dict(family='MLE',window=480));out['MLE480_strong']=strong
 return out,logs

def main():
 start=time.monotonic();dates,p=base.read_prefix();initial=int(np.sum(dates<'2020-01-01'))
 orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 if len(orig)!=451:raise ValueError('451 protocol')
 prep=prepare(p);true=p[orig[:,None]+H]-p[orig,None];years=np.array([d[:4] for d in dates[orig]])
 names=['RW','MLE_matched','residual_ridge','fixed_interaction','ESS_interaction','MLE480_strong'];pred={name:np.zeros_like(true) for name in names};logs=[]
 for fit in range(initial,len(p),40):
  if time.monotonic()-start>900:raise TimeoutError('frozen budget')
  pos=np.where((orig>=fit)&(orig<fit+40))[0]
  if not len(pos):continue
  out,l=block(p,fit,orig[pos],prep);logs+=l
  for name,value in out.items():pred[name][pos]=value
 results=[]
 for name in names:
  score,rows=base.score(pred[name],true,years);results.append(dict(name=name,score=score,by_year=rows));print(name,score,flush=True)
 np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',origins=orig,dates=dates[orig],true_changes=true,**pred)
 base.dump('ALL_RESULTS.json',results);base.dump('TRAINING_LOGS.json',logs)
 # Independent equivalence and suffix perturbation replay.
 fit=initial;oo=orig[(orig>=fit)&(orig<fit+40)][:3];first,_=block(p,fit,oo,prep)
 q=p.copy();q[oo.max()+1:]+=12345;second,_=block(q,fit,oo,prepare(q))
 error=max(float(np.max(np.abs(first[k]-second[k]))) for k in first)
 rng=np.random.default_rng(19);A=rng.normal(size=(120,44));A-=A.mean(0);Y=rng.normal(size=(120,3));B,lam,df=ridge(A,Y,3.)
 u,s,v=np.linalg.svd(A,full_matrices=False);ref=(v.T*(s/(s*s+lam)))@(u.T@Y);eq=float(np.max(np.abs(B-ref)))
 if error>1e-10 or eq>1e-10 or df>3.00000001:raise ValueError('verification failure')
 base.dump('VERIFICATION.json',dict(future_suffix_max_difference=error,svd_ridge_equivalence_max_difference=eq,cap_df=df,origins=len(orig),finite=all(np.isfinite(a).all() for a in pred.values()),maturity=all(a['max_label']<a['fit'] for a in logs),no_2022_read=True))
 base.dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-start,budget=900,configuration_count=len(names),development_only=True,novelty=False,reason='ordinary polynomial residual ridge with autocorrelation-heuristic penalty selection',single_pass=True))
if __name__=='__main__':main()
