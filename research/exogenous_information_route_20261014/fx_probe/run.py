import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib,datetime,time,sys,io
from pathlib import Path
import requests,numpy as np
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]; START=time.monotonic(); LOG=[]; CHECK=[]
sys.path.insert(0,str(ROOT/'research/deep_direction_review_20261013/calibration'))
import calibrate as base
base.OUT=OUT

def dump(n,v): (OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def require(c,m):
 if not c:raise ValueError(m)
def acquire():
 raw=OUT/'raw_LOCAL_ONLY';raw.mkdir(exist_ok=True);receipts=[];series=[]
 for cur in ['CNY','USD']:
  url=f'https://data-api.ecb.europa.eu/service/data/EXR/D.{cur}.EUR.SP00.A?startPeriod=2018-01-01&endPeriod=2021-12-31&format=csvdata'
  rec=dict(currency=cur,url=url,historical_first_available_at=None)
  try:
   with requests.get(url,timeout=(15,60),stream=True) as r:
    chunks=[];size=0
    for ch in r.iter_content(8192):
     chunks.append(ch);size+=len(ch);require(size<1500000,'response cap')
    b=b''.join(chunks);(raw/f'{cur}.csv').write_bytes(b)
    rec.update(status=r.status_code,final_url=r.url,headers=dict(r.headers),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),response_completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipts.append(rec);dump('ACQUISITION_RECEIPTS.json',receipts);require(r.status_code==200,'SOURCE BLOCKED STOP no aliases')
   rows=list(csv.DictReader(io.StringIO(b.decode('utf-8-sig')))); vals={}
   for a in rows:
    require(a['CURRENCY']==cur and a['CURRENCY_DENOM']=='EUR','wrong units')
    if a['OBS_VALUE']:vals[a['TIME_PERIOD']]=float(a['OBS_VALUE'])
   require(len(vals)>950,'insufficient ECB observations');series.append(vals)
  except Exception as e:
   rec.update(error=str(e),failed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump('ACQUISITION_RECEIPTS.json',receipts if rec in receipts else receipts+[rec]);raise
 keys=sorted(set(series[0])&set(series[1]));require(len(keys)==len(series[0])==len(series[1]),'unmatched dates')
 return np.array(keys,dtype='datetime64[D]'),np.log(np.array([series[0][k]/series[1][k] for k in keys]))
def fxfeatures(dates,fd,fv,lag):
 out=np.full((len(dates),2),np.nan);ix=np.searchsorted(fd,np.array(dates,dtype='datetime64[D]')-np.timedelta64(lag+1,'D'),side='right')-1
 for t,k in enumerate(ix):
  if k>=5:out[t]=[fv[k]-fv[k-1],fv[k]-fv[k-5]]
 return out,ix

def fit(X,Y,a):
 mu=X.mean(0);sd=np.maximum(X.std(0),1e-10);ym=Y.mean(0);A=(X-mu)/sd;B=np.linalg.solve(A.T@A+len(X)*a*np.eye(X.shape[1]),A.T@(Y-ym));return mu,sd,ym,B

def pred(s,X):mu,sd,ym,B=s;return ((X-mu)/sd)@B+ym

def choose(X,Y,idx,end,label,lag):
 boundary=idx[-120];tr=idx[idx+20<boundary];va=idx[-120:];vals=[]
 for a in [.1,10.]:
  s=fit(X[tr],Y[tr],a);loss=float(np.mean((pred(s,X[va])-Y[va])**2));vals.append(loss)
  LOG.append(dict(end=int(end),version=label,lag=lag,alpha=a,inner_mse=loss,inner_n=len(tr),validation_n=len(va),inner_max_target=int(tr[-1]+20),validation_first=int(boundary),outer_max_target=int(idx[-1]+20),outer_n=len(idx)))
 a=[.1,10.][int(np.argmin(vals))];return fit(X[idx],Y[idx],a),a

def main():
 fd,fv=acquire();dates,p=base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]]);require(len(orig)==451,'451 origins')
 px=np.zeros((len(p),4))
 for t in range(60,len(p)):px[t]=[p[t]-p[t-1],(p[t]-p[t-5])/5,(p[t]-p[t-20])/20,(p[t]-p[t-60])/60]
 Y=np.zeros((len(p),3));ii=np.arange(len(p)-20);Y[ii]=p[ii[:,None]+base.H]-p[ii,None];truth=Y[orig];years=np.array([s[:4] for s in dates[orig]]);saved=dict(origins=orig,truth=truth,years=years,prices=p,dates=dates,fx_dates=fd.astype(str),fx_log=fv,RW=np.zeros_like(truth));results=[]
 for lag in [2,3,5]:
  fx,ix=fxfeatures(dates,fd,fv,lag);X=np.column_stack([px,fx]);require(np.isfinite(fx[orig]).all(),'missing evaluation FX')
  outputs={k:np.zeros_like(truth) for k in ['priceonly','priceFX']}
  for end in range(initial,len(p),40):
   require(time.monotonic()-START<900,'900seconds exceeded');pos=np.where((orig>=end)&(orig<end+40))[0]
   if not len(pos):continue
   idx=np.arange(max(60,end-20-480),end-20);idx=idx[np.isfinite(fx[idx]).all(axis=1)];require(len(idx)>160,'training support');require(idx[-1]+20<end,'maturity')
   # Perturb all FX not admissible at first issuance; actually regenerate and refit.
   cutoff=np.datetime64(dates[end])-np.timedelta64(lag+1,'D');v2=fv.copy();v2[fd>cutoff]+=1000+np.arange(np.sum(fd>cutoff));f2,_=fxfeatures(dates,fd,v2,lag);X2=np.column_stack([px,f2])
   for name,A,A2 in [('priceonly',px,px),('priceFX',X,X2)]:
    s,a=choose(A,Y,idx,end,name,lag);outputs[name][pos]=pred(s,A[orig[pos]])
    s2,a2=choose(A2,Y,idx,end,name+'_futureFX_refit',lag);delta=max(float(np.max(np.abs(z-z2))) for z,z2 in zip(s,s2));pd=float(np.max(np.abs(pred(s,A[end:end+1])-pred(s2,A2[end:end+1]))));CHECK.append(dict(lag=lag,end=int(end),version=name,parameter_difference=delta,issuance_prediction_difference=pd,alpha_same=a==a2));require(delta<1e-10 and pd<1e-10 and a==a2,'futureFX refit failure')
  for name,pr in outputs.items():
   key=f'{name}_lag{lag}';saved[key]=pr;score,by=base.score(pr,truth,years);results.append(dict(name=key,score=score,by_year=by))
 np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved);dump('TRAINING_LOGS.json',LOG);dump('FUTURE_FX_REFIT.json',CHECK);dump('RESULTS.json',results);dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-START,ecb_observations=len(fd),training_records=len(LOG),refit_checks=len(CHECK),origins=len(orig),contract_sha256=hashlib.sha256((OUT/'CONTRACT.md').read_bytes()).hexdigest(),historical_vintage_certified=False));print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
