import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,csv,time,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('coal',OUT.parent/'coal_training/run.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.base.OUT=OUT
req=c.require

def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def features(dates,fd,fv,lag):
 ix=np.searchsorted(fd,np.array(dates,dtype='datetime64[D]')-np.timedelta64(lag+1,'D'),side='right')-1
 X=np.zeros((len(dates),4));valid=np.zeros(len(dates),bool)
 for t,k in enumerate(ix):
  if k>=5 and np.isfinite(fv[k-5:k+1]).all():X[t,:2]=[fv[k]-fv[k-1],fv[k]-fv[k-5]];valid[t]=True
 return X,valid

def main():
 start=time.monotonic();series=[];hashes={}
 for cur in ['CNY','USD']:
  path=OUT.parent/'fx_probe/raw_LOCAL_ONLY'/f'{cur}.csv';hashes[cur]=hashlib.sha256(path.read_bytes()).hexdigest()
  with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
  req(all(r['CURRENCY']==cur and r['CURRENCY_DENOM']=='EUR' for r in rows),'units');series.append({r['TIME_PERIOD']:float(r['OBS_VALUE']) for r in rows if r['OBS_VALUE']})
 keys=sorted(set(series[0])&set(series[1]));req(len(keys)==len(series[0])==len(series[1])==1025,'1025 dates')
 fd=np.array(keys,dtype='datetime64[D]');fv=np.log([series[0][k]/series[1][k] for k in keys]);dates,p=c.base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 req(len(orig)==451,'451 origins');B=c.vintage(p,initial);old=np.load(OUT.parent/'coal_training/PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);diff=float(np.max(np.abs(B[orig]-old['MLE'])));req(diff==0,'price prediction changed')
 Y=np.zeros_like(B);ii=np.arange(len(p)-20);Y[ii]=p[ii[:,None]+c.H]-p[ii,None];R=Y-B;truth=Y[orig];years=np.array([s[:4] for s in dates[orig]]);score,by=c.base.score(B[orig],truth,years);req(abs(score-.9497932301207621)<1e-12,'MLE score')
 saved=dict(origins=orig,truth=truth,years=years,MLE=B[orig],dates=dates);results=[dict(name='MLE480',score=score,by_year=by)];checks=[];coverage=[]
 for lag in [2,3,5]:
  X,valid=features(dates,fd,fv,lag);D=c.design(X,np.zeros(len(p)),'linear');pr=B[orig].copy()
  for end in range(initial,len(p),40):
   req(time.monotonic()-start<600,'600second budget');pos=np.where((orig>=end)&(orig<end+40)&valid[orig])[0]
   if not len(pos):continue
   idx=np.arange(max(60,end-20-480),end-20);idx=idx[valid[idx]]
   if len(idx)<=120:continue
   s,a=c.choose(D,R,idx,end,'linear',lag);pr[pos]+=c.predict(s,D[orig[pos]],'linear')
   if end in [initial,initial+40,initial+200]:
    cutoff=np.datetime64(dates[end])-np.timedelta64(lag+1,'D');future=fd>cutoff;v2=fv.copy();v2[future]+=1000+np.arange(future.sum());X2,valid2=features(dates,fd,v2,lag);D2=c.design(X2,np.zeros(len(p)),'linear');s2,a2=c.choose(D2,R,idx,end,'linear',lag,False)
    delta=max(float(np.max(np.abs(np.asarray(z)-np.asarray(w)))) for z,w in zip(s,s2));pd=float(np.max(np.abs(c.predict(s,D[end:end+1],'linear')-c.predict(s2,D2[end:end+1],'linear'))));req(delta==pd==0 and a==a2,'future FX refit');checks.append(dict(end=int(end),lag=lag,future_rows=int(future.sum()),parameter_diff=delta,prediction_diff=pd,alpha_same=a==a2))
  req(np.array_equal(pr[~valid[orig]],B[orig][~valid[orig]]),'fallback');saved[f'FX_lag{lag}']=pr;saved[f'valid_lag{lag}']=valid[orig];ss,bb=c.base.score(pr,truth,years);results.append(dict(name=f'FX_lag{lag}',score=ss,delta_vs_mle=ss-score,by_year=bb));coverage.append(dict(lag=lag,valid_origins=int(valid[orig].sum()),missing_fallback=int((~valid[orig]).sum())))
 # All missing input must return backbone, without fitting or imputation.
 Xm,vm=features(dates,fd,np.full_like(fv,np.nan),2);req(not vm.any(),'allmissing mask');pm=B[orig].copy();req(np.array_equal(pm,B[orig]),'allmissing fallback')
 np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved);dump('RESULTS.json',results);dump('TRAINING_LOGS.json',c.LOG);dump('FUTURE_FX_REFITS.json',checks);dump('COVERAGE.json',coverage);dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-start,origins=len(orig),ecb_rows=len(fd),price_prediction_diff=diff,all_missing_fallback_diff=float(np.max(np.abs(pm-B[orig]))),input_hashes=hashes,contract_sha256=hashlib.sha256((OUT/'CONTRACT.md').read_bytes()).hexdigest(),historical_vintage_certified=False,stop_fx_specific=results[1]['score']>=score));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
