import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SPEC=importlib.util.spec_from_file_location('historical_base',ROOT/'research/deep_direction_review_20261013/calibration/calibrate.py')
base=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(base);base.OUT=OUT
H=base.H
LOG=[];CHECK=[]
def require(c,msg):
 if not c:raise ValueError(msg)
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def features(dates,ed,v,lag):
 dd=np.array(dates,dtype='datetime64[D]');available=ed+np.timedelta64(lag,'D');ix=np.searchsorted(available,dd,side='right')-1
 X=np.zeros((len(dd),4));X[:,3]=1;valid=np.zeros(len(dd),bool);session_age=np.zeros(len(dd))
 for t,j in enumerate(ix):
  if j<1:continue
  age=int((dd[t]-available[j])/np.timedelta64(1,'D'));gap=int((available[j]-available[j-1])/np.timedelta64(1,'D'))
  if age>30 or gap>30 or str(ed[j])[:4]!=str(ed[j-1])[:4]:continue
  valid[t]=True;X[t]=[v[j],v[j]-v[j-1],age,0];session_age[t]=t-np.searchsorted(dd,available[j])
 return X,valid,session_age,ix

def design(X,age,name):
 if name=='linear':return np.repeat(X[:,None,:],3,axis=1)
 D=np.repeat((X[:,None,:]*H[None,:,None]),1,axis=0)
 k=lambda a:np.maximum(1-a/12,0)
 D[:,:,1]=X[:,1,None]*(k(age[:,None]+H)-k(age[:,None]))
 return D

def fit(D,R,idx,alpha,name):
 if name=='linear':
  A=D[idx,0];mu=A.mean(0);sd=np.maximum(A.std(0),1e-10);ym=R[idx].mean(0);Z=(A-mu)/sd
  B=np.linalg.solve(Z.T@Z+len(idx)*alpha*np.eye(4),Z.T@(R[idx]-ym));return mu,sd,ym,B
 A=D[idx].reshape(-1,4);y=R[idx].reshape(-1);mu=A.mean(0);sd=np.maximum(A.std(0),1e-10);ym=y.mean();Z=(A-mu)/sd
 B=np.linalg.solve(Z.T@Z+len(A)*alpha*np.eye(4),Z.T@(y-ym));return mu,sd,ym,B

def predict(s,D,name):
 mu,sd,ym,B=s
 return ((D[:,0]-mu)/sd)@B+ym if name=='linear' else ((D-mu)/sd)@B+ym

def choose(D,R,idx,end,name,lag,record=True):
 require(len(idx)>120,'too little common coal support');boundary=idx[-80];tr=idx[idx+20<boundary];va=idx[-80:];require(len(tr)>20,'purged inner support')
 losses=[]
 for a in [.1,10.]:
  s=fit(D,R,tr,a,name);losses.append(float(np.mean((predict(s,D[va],name)-R[va])**2)))
 a=[.1,10.][int(np.argmin(losses))]
 if record:LOG.append(dict(end=int(end),lag=lag,name=name,alpha=a,losses=losses,outer_n=len(idx),inner_n=len(tr),validation_n=80,inner_last_target=int(tr[-1]+20),validation_first=int(boundary),outer_last_target=int(idx[-1]+20)))
 require(tr[-1]+20<boundary and idx[-1]+20<end,'purge/maturity')
 return fit(D,R,idx,a,name),a

def vintage(p,initial):
 bank=base.filter_bank(p);B=np.zeros((len(p),3));schedule=list(range(60,initial,40))+list(range(initial,len(p),40))
 for j,end in enumerate(schedule):
  stop=schedule[j+1] if j+1<len(schedule) else len(p);orig=np.arange(end,stop)
  B[orig]=base.predict(p,end,orig,dict(family='MLE',window=480),dict(bank=bank))[0]
 return B

def main():
 begin=time.monotonic();LOG.clear();CHECK.clear();dates,p=base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 require(len(orig)==451,'451 origins');B=vintage(p,initial);truth=p[orig[:,None]+H]-p[orig,None];years=np.array([d[:4] for d in dates[orig]])
 mle,by=base.score(B[orig],truth,years);require(abs(mle-.9497932301207621)<1e-12,'MLE reproduction')
 Y=np.zeros_like(B);ii=np.arange(len(p)-20);Y[ii]=p[ii[:,None]+H]-p[ii,None];R=Y-B
 panel=OUT.parent/'coal_probe/coverage_completion'
 with (panel/'coal_5500_LITERAL_LOCAL_ONLY.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 with (panel/'announcement_index_LOCAL_ONLY.csv').open(encoding='utf-8-sig') as f:index=list(csv.DictReader(f))
 require({(r['url'],r['source_sha256'],r['publication_displayed']) for r in rows}=={(r['url'],r['source_sha256'],r['publication_displayed']) for r in index},'index identity')
 require(len(rows)==len(index)==123,'panel count');require(all(r['is_5500_literal']=='True' and r['unit_literal']=='吨' and ''.join(r['product_spec_literal'].split())=='山西优混（5500大卡）' for r in rows),'literal spec')
 rows.sort(key=lambda r:r['publication_displayed']);ed=np.array([r['publication_displayed'][:10] for r in rows],dtype='datetime64[D]');v=np.array([float(r['price_cny_per_tonne']) for r in rows])
 saved=dict(origins=orig,truth=truth,years=years,dates=dates,MLE=B[orig]);results=[dict(name='MLE480',score=mle,by_year=by)];coverage=[]
 for lag in [1,2]:
  X,valid,age,ix=features(dates,ed,v,lag);coverage.append(dict(lag=lag,valid_origins=int(valid[orig].sum()),fallback_origins=int((~valid[orig]).sum()),year_valid={str(y):int(valid[orig[years==y]].sum()) for y in set(years)}));saved['features_lag'+str(lag)]=X;saved['valid_lag'+str(lag)]=valid;saved['age_sessions_lag'+str(lag)]=age
  for name in ['linear','kernel']:
   D=design(X,age,name);pr=B[orig].copy()
   for end in range(initial,len(p),40):
    require(time.monotonic()-begin<900,'budget');pos=np.where((orig>=end)&(orig<end+40)&valid[orig])[0]
    if not len(pos):continue
    idx=np.arange(max(60,end-20-480),end-20);idx=idx[valid[idx]];s,a=choose(D,R,idx,end,name,lag);pr[pos]+=predict(s,D[orig[pos]],name)
    vv=v.copy();future=ed+np.timedelta64(lag,'D')>np.datetime64(dates[end]);vv[future]+=10000+np.arange(future.sum());X2,valid2,age2,_=features(dates,ed,vv,lag);D2=design(X2,age2,name);s2,a2=choose(D2,R,idx,end,name,lag,False)
    delta=max(float(np.max(np.abs(np.asarray(z)-np.asarray(z2)))) for z,z2 in zip(s,s2));pd=float(np.max(np.abs(predict(s,D[end:end+1],name)-predict(s2,D2[end:end+1],name))))
    require(delta<1e-12 and pd<1e-12 and a==a2,'future coal refit');CHECK.append(dict(kind='future_coal',end=int(end),lag=lag,name=name,param_diff=delta,pred_diff=pd,future_events=int(future.sum())))
    if end in [459,499,659]:
     pp=p.copy();pp[end+1:]+=10000+np.arange(len(p)-end-1);BB=vintage(pp,initial);YY=np.zeros_like(B);YY[ii]=pp[ii[:,None]+H]-pp[ii,None];RR=YY-BB;sp,ap=choose(D,RR,idx,end,name,lag,False)
     dp=max(float(np.max(np.abs(np.asarray(z)-np.asarray(z2)))) for z,z2 in zip(s,sp));forecastdiff=float(np.max(np.abs(B[end]+predict(s,D[end:end+1],name)[0]-BB[end]-predict(sp,D[end:end+1],name)[0])))
     require(dp<1e-12 and forecastdiff<1e-12 and a==ap,'future price refit');CHECK.append(dict(kind='future_price',end=int(end),lag=lag,name=name,param_diff=dp,pred_diff=forecastdiff))
   require(np.array_equal(pr[~valid[orig]],B[orig][~valid[orig]]),'fallback not exact');score,by=base.score(pr,truth,years);key=f'{name}_lag{lag}';saved[key]=pr;results.append(dict(name=key,score=score,delta_vs_mle=score-mle,by_year=by))
 np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved);dump('RESULTS.json',results);dump('TRAINING_LOGS.json',LOG);dump('CAUSAL_REFITS.json',CHECK);dump('COVERAGE.json',coverage)
 receipt=dict(origins=len(orig),mle_score=mle,checks_passed=True,seconds=time.monotonic()-begin,historical_first_version_certified=False,events=123,last_event=str(ed[-1]),price_last=dates[-1],refit_checks=len(CHECK),contract_sha256=hashlib.sha256((OUT/'CONTRACT.md').read_bytes()).hexdigest())
 dump('RUN_RECEIPT.json',receipt);print(json.dumps(results,ensure_ascii=False,indent=2));return receipt
if __name__=='__main__':main()
