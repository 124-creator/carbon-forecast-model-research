import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
import importlib.util,json,csv,time,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
SRC=Path('E:/碳价格/独立数据归档/carbon-forecast-model-research/research/exogenous_information_route_20261014')
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('coal',SRC/'coal_training/run.py');fx=load('fx',SRC/'fx_strong_probe/run.py');c.base.OUT=OUT
LOG=[];FITS=0
def dump(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def req(b,s):
 if not b: raise ValueError(s)
def state(p):
 Q=np.zeros((8,60));start=0
 for j,w in enumerate([1,1,1,2,5,10,20,20]): Q[j,start:start+w]=1/np.sqrt(w);start+=w
 return c.base.raw_features(p)@Q.T
def fit(A,Y,a):
 global FITS
 FITS+=1
 return np.linalg.solve(A.T@A+len(A)*a*np.eye(A.shape[1]),A.T@Y)
def norm(P,idx):
 mu=P[idx].mean(0);sd=np.maximum(P[idx].std(0),1e-10)
 return np.column_stack([np.ones(len(P)),(P-mu)/sd]),mu,sd
def block(P,R,S,idx,end,record=True):
 boundary=idx[-80];tr=idx[idx+20<boundary];va=idx[-80:]
 req(len(tr)>20 and tr[-1]+20<boundary and idx[-1]+20<end,'purge/maturity')
 A,mu,sd=norm(P,tr);pl=[float(np.mean(R[va]**2))];pb=[]
 for a in [.1,10.]:
  b=fit(A[tr],R[tr],a);pb.append(b);pl.append(float(np.mean((R[va]-A[va]@b)**2)))
 pi=int(np.argmin(pl));pa=[None,.1,10.][pi];bt=np.zeros((9,3)) if pa is None else pb[pi-1]
 rt=R[tr]-A[tr]@bt;rv=R[va]-A[va]@bt
 chosen=[];inner=[]
 for X in S:
  xm=X[tr].mean(0);xs=np.maximum(X[tr].std(0),1e-10);Z=(X-xm)/xs
  losses=[float(np.mean(rv**2))]
  for a in [.1,10.]:
   proj=fit(A[tr],Z[tr],a);U=Z-A@proj;b=fit(U[tr],rt,a);losses.append(float(np.mean((rv-U[va]@b)**2)))
  ai=int(np.argmin(losses));chosen.append([None,.1,10.][ai]);inner.append(losses)
 # Refit selected nuisance/calibration on identical full mature pool.
 A,mu,sd=norm(P,idx);b=np.zeros((9,3)) if pa is None else fit(A[idx],R[idx],pa)
 params=[mu,sd,b];price=A@b;pred=[price]
 for X,a in zip(S,chosen):
  xm=X[idx].mean(0);xs=np.maximum(X[idx].std(0),1e-10);Z=(X-xm)/xs
  proj=np.zeros((9,X.shape[1]));beta=np.zeros((X.shape[1],3))
  if a is not None:
   proj=fit(A[idx],Z[idx],a);U=Z-A@proj;beta=fit(U[idx],R[idx]-price[idx],a)
  pred.append(price+(Z-A@proj)@beta);params.extend([xm,xs,proj,beta])
 if record: LOG.append(dict(end=int(end),n=len(idx),inner_n=len(tr),validation_n=len(va),max_target=int(idx[-1]+20),boundary=int(boundary),price_alpha=pa,source_alpha=chosen,price_losses=pl,source_losses=inner,coal_unique_values=int(len(np.unique(S[0][idx],axis=0)))))
 return pred,params,[pa]+chosen

def main():
 start=time.monotonic();dates,p=c.base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 req(len(orig)==451,'451');B=c.vintage(p,initial);old=np.load(SRC/'coal_training/PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);req(np.array_equal(B[orig],old['MLE']),'MLE replay')
 rows=list(csv.DictReader((SRC/'coal_probe/coverage_completion/coal_5500_LITERAL_LOCAL_ONLY.csv').open(encoding='utf-8-sig')));req(len(rows)==123,'123 events');rows.sort(key=lambda r:r['publication_displayed']);ed=np.array([r['publication_displayed'][:10] for r in rows],dtype='datetime64[D]');v=np.array([float(r['price_cny_per_tonne']) for r in rows])
 series=[]
 for cur in ['CNY','USD']:
  rr=list(csv.DictReader((SRC/f'fx_probe/raw_LOCAL_ONLY/{cur}.csv').open(encoding='utf-8-sig')));series.append({r['TIME_PERIOD']:float(r['OBS_VALUE']) for r in rr if r['OBS_VALUE']})
 keys=sorted(set(series[0])&set(series[1]));req(len(keys)==1025,'FX1025');fd=np.array(keys,dtype='datetime64[D]');fv=np.log([series[0][k]/series[1][k] for k in keys])
 X,vc,_,_=c.features(dates,ed,v,1);XF,vf=fx.features(dates,fd,fv,2);S=[X[:,1:2],XF[:,:2]];P=state(p);Y=np.zeros_like(B);ii=np.arange(len(p)-20);Y[ii]=p[ii[:,None]+c.H]-p[ii,None];R=Y-B
 predictions=[B[orig].copy() for _ in range(3)];checks=[]
 for end in range(initial,len(p),40):
  req(time.monotonic()-start<840,'14minute run limit');pos=np.where((orig>=end)&(orig<end+40))[0]
  if not len(pos):continue
  idx=np.arange(max(60,end-20-480),end-20);idx=idx[vc[idx]&vf[idx]];req(len(idx)>120,'common support')
  pr,params,sel=block(P,R,S,idx,end)
  for j in range(3):
   cc=pr[j][orig[pos]].copy()
   if j: cc[~[vc,vf][j-1][orig[pos]]]=pr[0][orig[pos]][~[vc,vf][j-1][orig[pos]]]
   predictions[j][pos]+=cc
  if end in [initial,initial+40,initial+200]:
   for kind in ['coal','fx','price']:
    pp=p.copy();vv=v.copy();ff=fv.copy()
    if kind=='coal': mask=ed+np.timedelta64(1,'D')>np.datetime64(dates[end]);vv[mask]+=10000+np.arange(mask.sum())
    if kind=='fx': mask=fd>np.datetime64(dates[end])-np.timedelta64(3,'D');ff[mask]+=1000+np.arange(mask.sum())
    if kind=='price': pp[end+1:]+=10000+np.arange(len(pp)-end-1)
    XX,vcc,_,_=c.features(dates,ed,vv,1);XXF,vff=fx.features(dates,fd,ff,2);BB=c.vintage(pp,initial) if kind=='price' else B;YY=np.zeros_like(B);YY[ii]=pp[ii[:,None]+c.H]-pp[ii,None];PP=state(pp);SS=[XX[:,1:2],XXF[:,:2]]
    ix=np.arange(max(60,end-20-480),end-20);ix=ix[vcc[ix]&vff[ix]];req(np.array_equal(idx,ix),'support invariant')
    pr2,pars2,sel2=block(PP,YY-BB,SS,ix,end,False);pd=max(float(np.max(np.abs(a-b))) for a,b in zip(params,pars2));dd=max(float(np.max(np.abs(B[end]+a[end]-BB[end]-b[end]))) for a,b in zip(pr,pr2));req(pd<1e-12 and dd<1e-12 and sel==sel2,'future actual refit');checks.append(dict(kind=kind,end=int(end),parameter_diff=pd,prediction_diff=dd,selection_same=sel==sel2))
 for j,valid in enumerate([vc,vf],1): req(np.array_equal(predictions[j][~valid[orig]],predictions[0][~valid[orig]]),'matched price fallback')
 saved=dict(origins=orig,truth=Y[orig],years=np.array([d[:4] for d in dates[orig]]),MLE=B[orig],PRICE=predictions[0],COAL=predictions[1],FX=predictions[2],valid_coal=vc[orig],valid_fx=vf[orig]);np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved)
 dump('TRAINING_LOGS.json',LOG);dump('FUTURE_REFITS.json',checks);dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-start,actual_ridge_solves=FITS,events=123,origins=451,refreshes=len(LOG),historical_strongstate=.942705,contract_sha256=hashlib.sha256((OUT/'CONTRACT.md').read_bytes()).hexdigest(),sequential_not_joint=True,coal_missing=int((~vc[orig]).sum()),fx_missing=int((~vf[orig]).sum())))
 print(json.dumps(dict(seconds=time.monotonic()-start,solves=FITS,checks=len(checks))))
if __name__=='__main__':main()
