"""Bounded rejection experiment; only synthetic archive price arrays are opened."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.linalg import solve
import pathlib,json,time,traceback,hashlib
BASE=pathlib.Path(__file__).resolve().parent
SCR=pathlib.Path('C:/Users/Dreamboat/AppData/Local/hermes/cache/scratch/horizon_probe_20261013')
H=np.array([1,5,20])
def solve_head(A,y,C,d,alpha,lam,dual=False):
 R=alpha*np.eye(A.shape[1])+lam*C.T@C
 q=lam*C.T@d
 w=solve(A.T@A+R,A.T@y+q,assume_a='pos')
 if not dual:return w,None
 w0=solve(R,q,assume_a='pos'); RA=solve(R,A.T,assume_a='pos')
 wd=w0+RA@solve(np.eye(len(A))+A@RA,y-A@w0,assume_a='pos')
 return w,wd

def features(p):
 r=np.r_[0,np.diff(p)]; raw=np.zeros((len(p),60))
 for t in range(60,len(p)):raw[t]=r[t-59:t+1][::-1]
 blocks=[1,1,1,2,5,10,20,20];x=[];a=0
 for b in blocks:x.append(raw[:,a:a+b].sum(1)/np.sqrt(b));a+=b
 x=np.array(x).T
 # Fixed random nonlinear features, no generator poles/directions/labels.
 rng=np.random.default_rng(917);v=rng.normal(size=(8,12))/np.sqrt(8)
 return np.c_[x,np.tanh(x@v),np.ones(len(x))]
def target(p,idx):return p[idx[:,None]+H]-p[idx,None]
def design(f,p,idx,hs):
 n=len(idx);d=f.shape[1];A=np.zeros((n*3,d*3))
 for j in range(3):A[j*n:(j+1)*n,j*d:(j+1)*d]=f[idx]/hs[j]/np.sqrt(n*3)
 y=(target(p,idx)/hs).T.ravel()/np.sqrt(n*3)
 C=[];ds=[]
 for j,l in [(1,0),(2,1)]:
  k=int(H[j]-H[l]);q=idx[idx+k+H[l]<idx[-1]+21]
  c=np.zeros((len(q),d*3));c[:,j*d:(j+1)*d]=f[q]/hs[j];c[:,l*d:(l+1)*d]=-f[q+k]/hs[j]
  C.append(c);ds.append((p[q+k]-p[q])/hs[j])
 C=np.concatenate(C);ds=np.concatenate(ds);C/=np.sqrt(len(ds));ds/=np.sqrt(len(ds))
 return A,y,C,ds

def run():
 start=time.monotonic();records=[];configs=[];failures=[];audit=[]
 paths=sorted((SCR/'F/carbon_stageF_20261013/outputs/synthetic_first/archives').glob('*.npz'))+sorted((SCR/'G/carbon_stageG_20261013/outputs/synthetic_first/archives').glob('*.npz'))
 for path in paths:
  if time.monotonic()-start>850:raise TimeoutError('Compute budget safety stop')
  try:
   with np.load(path,allow_pickle=False) as z:
    key='price' if 'price' in z.files else 'p'
    p=np.array(z[key],float)
   if p.ndim!=1 or not np.isfinite(p).all():raise ValueError('invalid prices')
   audit.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'read_array':key,'n':len(p),'synthetic_only':True})
   # 240 mature train origins, 240 tuning origins; embargo before evaluation.
   tr=np.arange(60,300);va=np.arange(340,580);ev=np.arange(620,len(p)-20)
   hs=np.sqrt(np.mean(target(p,tr)**2,axis=0));f=features(p)
   scale=f[tr].std(0);scale[scale<1e-10]=1;f=f/scale
   A,y,C,d=design(f,p,tr,hs);fv=f[va];yv=target(p,va)
   candidates=[]
   for alpha in [.001,.01,.1,1.]:
    w,_=solve_head(A,y,C,d,alpha,0);B=w.reshape(3,-1).T
    # Exact reduced-rank ridge: truncate fitted values in ridge-whitened space.
    G=A[:len(tr),:f.shape[1]].T@A[:len(tr),:f.shape[1]]
    # Differing horizon scales: whiten design of standardized-output common ridge below.
    X=f[tr]/np.sqrt(len(tr));Y=target(p,tr)/hs/np.sqrt(len(tr));T=X.T@X+alpha*np.eye(f.shape[1]);L=np.linalg.cholesky(T)
    M=solve(L,X.T@Y,assume_a='gen');u,s,vh=np.linalg.svd(M,full_matrices=False)
    for rank in [1,2,3]:
     br=solve(L.T,(u[:,:rank]*s[:rank])@vh[:rank],assume_a='gen')*hs
     candidates.append(('LOWRANK',alpha,rank,br))
    candidates.append(('RIDGE',alpha,0,B))
    for lam in [.01,.1,1.,10.]:
     w,_=solve_head(A,y,C,d,alpha,lam);candidates.append(('CONSISTENCY',alpha,lam,w.reshape(3,-1).T))
   best={}
   for fam,alpha,param,B in candidates:
    loss=float(np.mean(((fv@B-yv)/hs)**2));row={'path':path.stem,'stage':path.parts[-5],'family':fam,'alpha':alpha,'parameter':param,'tune_loss':loss};configs.append(row)
    if fam not in best or loss<best[fam][0]:best[fam]=(loss,alpha,param,B)
   preds={'RW':np.zeros((len(ev),3))}; selected={}
   for fam,(_,alpha,param,B) in best.items():preds[fam]=f[ev]@B;selected[fam]={'alpha':alpha,'parameter':param}
   # Exact augmented fixed-feature regression counterpart for selected consistency.
   alpha=selected['CONSISTENCY']['alpha'];lam=selected['CONSISTENCY']['parameter']
   aug=np.concatenate([A,np.sqrt(lam)*C,np.sqrt(alpha)*np.eye(A.shape[1])]);yy=np.r_[y,np.sqrt(lam)*d,np.zeros(A.shape[1])]
   wk=np.linalg.lstsq(aug,yy,rcond=None)[0];preds['FIXED_AUGMENTED_KERNEL']=f[ev]@wk.reshape(3,-1).T
   err=float(np.max(abs(preds['CONSISTENCY']-preds['FIXED_AUGMENTED_KERNEL'])))
   ye=target(p,ev)
   for fam,pr in preds.items():
    rr=np.mean((pr-ye)**2,0)/np.mean(ye**2,0)
    records.append({'path':path.stem,'stage':path.parts[-5],'family':fam,'relative_mse':float(rr.mean()),'relative_mse_h':rr.tolist(),'selected':selected.get(fam),'max_kernel_equivalence_error':err,'train_count':len(tr),'tune_count':len(va),'eval_count':len(ev),'max_train_label':int(tr[-1]+20),'first_tune_origin':int(va[0]),'max_tune_label':int(va[-1]+20),'first_eval_origin':int(ev[0])})
   np.savez_compressed(BASE/('predictions_'+path.parts[-5]+'_'+path.stem+'.npz'),origins=ev,target=ye,**preds)
   print('completed',path.stem,'equivalence',err,flush=True)
  except Exception as e:
   failures.append({'path':str(path),'error':repr(e),'traceback':traceback.format_exc()});print('FAILED',path,e,flush=True)
  for name,obj in [('all_configurations.json',configs),('selected_results.json',records),('failures.json',failures),('input_audit.json',audit)]:
   (BASE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
 summary={}
 for stage in sorted(set(r['stage'] for r in records)):
  summary[stage]={}
  for proc in sorted(set(r['path'].rsplit('_',1)[0] for r in records if r['stage']==stage)):
   rows=[r for r in records if r['stage']==stage and r['path'].rsplit('_',1)[0]==proc]
   summary[stage][proc]={fam:float(np.mean([r['relative_mse'] for r in rows if r['family']==fam])) for fam in sorted(set(r['family'] for r in rows))}
 result={'status':'REJECT_NOVELTY_FIXED_FEATURE_QUADRATIC_EQUIVALENCE','summary':summary,'paths_requested':len(paths),'paths_completed':len(records)//5,'records':len(records),'configurations':len(configs),'failures':failures,'seconds':time.monotonic()-start,'real_market_opened':False,'refit_after_tuning':False,'old_synthetic_development_not_confirmation':True}
 (BASE/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
