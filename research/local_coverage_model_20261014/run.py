import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,importlib.util
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'research/deep_direction_review_20261013/calibration'))
import calibrate as base
spec=importlib.util.spec_from_file_location('state',ROOT/'research/multi_angle_model_search_20261013/state/run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);base.OUT=OUT
P=json.loads((OUT/'PROTOCOL.json').read_text());START=time.monotonic();LOG=[];CHECK=[]
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def require(c,m):
 if not c:raise ValueError(m)
def raw(x,s):
 z=(x-s['mu'])/s['sd'];L=np.column_stack([np.ones(len(x)),z]);parts=[]
 for j,k in enumerate(s['knots']):
  zz=np.clip(z[:,j],k[0],k[-1]);parts.append(BSpline.design_matrix(zz,k,1).toarray())
 return L,np.column_stack(parts)
def design(x,s):
 L,B=raw(x,s);N=(B-L@s['projection'])/s['nscale'];return np.column_stack([L,N])
def fit(x,y,f,a):
 s={'mu':x.mean(0),'sd':np.maximum(x.std(0),1e-8)};z=(x-s['mu'])/s['sd'];ks=[]
 for j in range(x.shape[1]):
  inner=np.unique(np.quantile(z[:,j],P['knots_quantiles']));lo=min(float(z[:,j].min()),float(inner[0]))-1e-6;hi=max(float(z[:,j].max()),float(inner[-1]))+1e-6;ks.append(np.r_[lo,lo,inner,hi,hi])
 s['knots']=ks;L,B=raw(x,s);s['projection']=np.linalg.lstsq(L,B,rcond=1e-10)[0];N=B-L@s['projection'];s['nscale']=np.maximum(N.std(0),1e-6)
 mass=np.mean(B*B,0);inv=1/np.maximum(mass,1/len(x));inv=inv/inv.mean();q=len(inv);D=np.zeros((sum(len(k)-4 for k in ks),q));row=0;off=0
 for k in ks:
  n=len(k)-2
  for j in range(n-2):D[row,off+j:off+j+3]=[1,-2,1];row+=1
  off+=n
 C=np.diag(inv)+D.T@D;C=C/np.mean(np.diag(C));pen=np.eye(4+q);pen[0,0]=0
 if f=='coverage':pen[4:,4:]=C
 if f=='coverage_constant':pen[4:,4:]=np.eye(q)
 if f=='coverage_permuted':
  ix=np.arange(q)[::-1];pen[4:,4:]=C[np.ix_(ix,ix)]
 A=L if f=='linear' else design(x,s)
 if f=='linear':pen=pen[:4,:4]
 G=A.T@A+len(x)*a*pen;rhs=A.T@y;coef=solve(G,rhs,assume_a='pos');s.update(coef=coef,family=f,mass=mass,penalty=pen)
 # Independently solve augmented least squares, including semidefinite intercept penalty.
 ev,U=np.linalg.eigh(pen);R=np.sqrt(np.maximum(ev,0))[:,None]*U.T
 alt=np.linalg.lstsq(np.vstack([A,np.sqrt(len(x)*a)*R]),np.r_[y,np.zeros(len(pen))],rcond=1e-10)[0]
 s['audit']={'gradient':float(np.linalg.norm(G@coef-rhs)/len(x)),'augmented_prediction_difference':float(np.max(np.abs(A@(coef-alt)))),'n':len(x),'parameters':len(coef),'mass_min':float(mass.min()),'mass_max':float(mass.max())};return s
def predict(s,x):return (raw(x,s)[0] if s['family']=='linear' else design(x,s))@s['coef']
def train(x,y,idx,h,fit_end,label):
 require(time.monotonic()-START<P['budget_seconds'],'600 second compute budget');cut=len(idx)-P['validation'];boundary=int(idx[cut]);tr=idx[idx+h<boundary];va=idx[cut:];require(len(tr)>20,'inner support');require(tr[-1]+h<va[0],'purged validation');res={}
 for f in P['families']:
  vals=[];models=[]
  for a in P['alphas']:
   inner=fit(x[tr],y[tr],f,a);v=float(np.mean((predict(inner,x[va])-y[va])**2));s=fit(x[idx],y[idx],f,a);models.append(s);vals.append(v)
   LOG.append(dict(label=label,h=int(h),fit_end=int(fit_end),family=f,alpha=a,validation_mse=v,min_origin=int(idx[0]),max_origin=int(idx[-1]),max_target=int(idx[-1]+h),inner_max_target=int(tr[-1]+h),validation_first_origin=int(va[0]),knots=[k.tolist() for k in s['knots']],mass=s['mass'].tolist(),**s['audit']))
  res[f]=(models,int(np.argmin(vals)))
 return res

def synthetic():
 rows=[]
 for seed in P['synthetic_seeds']:
  rng=np.random.default_rng(seed);n=950;x=rng.normal(size=(n,3));x[:480,0]=rng.normal(-.8,.5,480);x[480:,0]=rng.normal(.8,.7,n-480)
  for task in ['coverage_shift_signal','null']:
   y=(2*np.maximum(x[:,0],0)+.6*x[:,1]**2 if task!='null' else np.zeros(n))+rng.normal(size=n)*.5
   idx=np.arange(120,460);rr=train(x,y,idx,20,480,'synthetic_'+task+'_'+str(seed));out={}
   for f,(ss,best) in rr.items():out[f]=[float(np.mean((predict(s,x[480:])-y[480:])**2)/np.mean(y[480:]**2)) for s in ss]
   rows.append(dict(seed=seed,task=task,ratios_rw=out))
 dump('SYNTHETIC.json',rows);return rows

def market():
 dates,p=base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]]);require(len(orig)==451,'451 origins');m,st=old.history(p,initial);truth=p[orig[:,None]+base.H]-p[orig,None];years=np.array([d[:4] for d in dates[orig]])
 names=[f+'_'+str(a) for f in P['families'] for a in P['alphas']]+[f+'_selected' for f in P['families']];pred={n:np.zeros_like(truth) for n in names};pred['MLE']=m[orig];pred['RW']=np.zeros_like(truth);saved=dict(origins=orig,truth=truth,years=years,origin_prices=p[orig],issuance_mle=m,states=st)
 for end in range(initial,len(p),40):
  pos=np.where((orig>=end)&(orig<end+40))[0]
  if not len(pos):continue
  for j,h in enumerate(base.H):
   x=np.column_stack([m[:,j],st]);y=np.zeros(len(p));avail=np.arange(len(p)-h);y[avail]=p[avail+h]-p[avail]-m[avail,j];idx=np.arange(max(120,end-int(h)-480),end-int(h));require(idx[-1]+h<end,'outer maturity');rr=train(x,y,idx,int(h),end,'market')
   for f,(ss,best) in rr.items():
    for a,s in zip(P['alphas'],ss):pred[f+'_'+str(a)][pos,j]=m[orig[pos],j]+predict(s,x[orig[pos]])
    pred[f+'_selected'][pos,j]=m[orig[pos],j]+predict(ss[best],x[orig[pos]])
   # Prefix replay and future-price perturbation regenerate issuance features, labels and fits.
   q=p.copy();q[end:]+=123+np.arange(len(p)-end)*1000;mq,sq=old.history(q,initial);mp,sp=old.history(p[:end],initial)
   CHECK.append(dict(end=end,h=int(h),feature_prefix_difference=float(np.max(np.abs(np.column_stack([mp[:,j],sp])-x[:end]))),future_feature_difference=float(np.max(np.abs(np.column_stack([mq[:end,j],sq[:end]])-x[:end]))),max_target=int(idx[-1]+h)))
   yq=q[idx+h]-q[idx]-mq[idx,j];require(np.array_equal(yq,y[idx]),'future target invariance')
   for f,(ss,best) in rr.items():
    for a,s in zip(P['alphas'],ss):
     alt=fit(np.column_stack([mq[idx,j],sq[idx]]),yq,f,a);diff=float(np.max(np.abs(predict(s,x[end-1:end])-predict(alt,x[end-1:end]))));CHECK.append(dict(end=end,h=int(h),family=f,alpha=a,prediction_future_difference=diff))
 rows=[]
 for n,pr in pred.items():
  value,by=base.score(pr,truth,years);rows.append(dict(name=n,score=value,by_year=by));saved[n]=pr
 np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved);dump('RESULTS.json',rows);return rows
if __name__=='__main__':
 syn=synthetic();rows=market();dump('TRAINING_LOGS.json',LOG);dump('CAUSAL_AUDIT.json',CHECK)
 require(max(r['gradient'] for r in LOG)<1e-7,'normal equations');require(max(r['augmented_prediction_difference'] for r in LOG)<1e-6,'augmented equivalence');require(all(max([v for k,v in r.items() if 'difference' in k] or [0])<1e-8 for r in CHECK),'causal differences')
 dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-START,training_records=len(LOG),results=len(rows),development_only=True,read_boundary=False,budget_seconds=600,diagnostic_only=True));print(json.dumps([(r['name'],r['score']) for r in rows],indent=2))
