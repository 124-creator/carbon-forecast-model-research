"""Bounded rolling SSA baseline. No novelty claim; no old model execution."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import solve
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'deep_direction_review_20261013/calibration'))
import calibrate as base
base.OUT=OUT
H=base.H

def feature_at(p,t):
 if t<59 or t>=len(p):raise ValueError('issuance index outside available prefix')
 x=np.asarray(p[t-59:t+1],float)
 if not np.isfinite(x).all():raise ValueError('nonfinite input')
 x=x-x[-1];L=16
 A=np.lib.stride_tricks.sliding_window_view(x,L).T
 u,s,v=np.linalg.svd(A,full_matrices=False)
 groups=np.zeros((3,60));counts=np.convolve(np.ones(L),np.ones(A.shape[1]))
 for j in range(L):
  freq=np.argmax(np.abs(np.fft.rfft(u[:,j]))) / L
  g=0 if freq<=.0625 else (1 if freq<=.1875 else 2)
  C=s[j]*np.outer(u[:,j],v[j])
  for i in range(L):groups[g,i:i+A.shape[1]]+=C[i]/counts[i:i+A.shape[1]]
 return np.diff(x)[::-1],np.diff(groups,axis=1)[:,::-1].reshape(-1)

def indices(fit):
 idx=np.arange(max(60,fit-20-480),fit-20)
 if len(idx)<40 or idx[-1]+20>=fit:raise ValueError('immature labels')
 return idx

def learned(p,X,fit,orig,c):
 idx=indices(fit);A=X[idx];mu=A.mean(0);sd=A.std(0);sd[sd<1e-12]=1
 A=(A-mu)/sd;T=(X[orig]-mu)/sd
 if c['backbone']=='RFF':
  rng=np.random.default_rng(c['seed']);W=rng.normal(size=(A.shape[1],48))/np.sqrt(A.shape[1]);phase=rng.uniform(0,2*np.pi,48)
  A=np.sqrt(2/48)*np.cos(A@W+phase);T=np.sqrt(2/48)*np.cos(T@W+phase)
 Y=p[idx[:,None]+H]-p[idx,None];ym=Y.mean(0);am=A.mean(0);A=A-am;T=T-am
 B=solve(A.T@A+c['alpha']*len(idx)*np.eye(A.shape[1]),A.T@(Y-ym),assume_a='pos')
 pred=T@B+ym
 return pred,dict(fit=fit,min_origin=int(idx[0]),max_origin=int(idx[-1]),max_label=int(idx[-1]+20),training_rows=len(idx),input_dim=X.shape[1],readout_dim=A.shape[1])

def matched_mle(p,fit,orig):
 idx=indices(fit);ll=[]
 # Identical training-origin support; only one-step target per mature origin.
 for s in idx:
  _,a,pars=base.filter_bank(p[s-59:s+2]);ll.append(a[-1])
 a=np.sum(ll,0);j=int(np.argmin(a[:,0]+len(idx)*np.log(np.maximum(a[:,1]/len(idx),1e-14))))
 preds=[]
 for t in orig:
  bank,_,_=base.filter_bank(p[t-59:t+1]);preds.append(bank[-1,j])
 return np.array(preds),dict(fit=fit,min_origin=int(idx[0]),max_origin=int(idx[-1]),max_label=int(idx[-1]+1),training_rows=len(idx),selected_grid=j)

def configs():
 cs=[dict(name='RW',representation='NONE',backbone='RW'),dict(name='MLE_matched',representation='NONE',backbone='MLE')]
 for rep in ['RAW','SSA']:
  for a in [.1,1.,10.]:
   cs.append(dict(name=f'{rep}_Ridge_a{a}',representation=rep,backbone='Ridge',alpha=a))
   for seed in [17,29]:cs.append(dict(name=f'{rep}_RFF_a{a}_s{seed}',representation=rep,backbone='RFF',alpha=a,seed=seed))
 return cs

def main():
 start=time.monotonic();cs=configs()
 base.dump('FROZEN_CONFIG.json',dict(configurations=cs,window=60,ssa_L=16,frequency_cutoffs=[.0625,.1875],training_window=480,refresh=40,seeds=[17,29],rff_nodes=48,horizons=H.tolist(),budget_seconds=900,stop='Complete frozen 20 configurations or explicit timeout; no adaptive search',selection='Development-only; average losses across seeds, not forecasts',budget_not_flops=True))
 dates,p=base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 if len(orig)!=451:raise ValueError('451 protocol')
 raw=np.zeros((len(p),59));ssa=np.zeros((len(p),177))
 for t in range(59,len(p)):
  if time.monotonic()-start>900:raise TimeoutError('900-second budget')
  raw[t],ssa[t]=feature_at(p,t)
 np.savez_compressed(OUT/'VINTAGES_LOCAL_ONLY.npz',issuance=np.arange(len(p)),earliest=np.maximum(0,np.arange(len(p))-59),valid=np.arange(len(p))>=59,RAW=raw,SSA=ssa)
 true=p[orig[:,None]+H]-p[orig,None];years=np.array([d[:4] for d in dates[orig]]);saved=dict(origins=orig,dates=dates[orig],true_changes=true);results=[];logs=[]
 for c in cs:
  pred=np.full(true.shape,np.nan)
  for fit in range(initial,len(p),40):
   if time.monotonic()-start>900:raise TimeoutError('900-second budget')
   pos=np.where((orig>=fit)&(orig<fit+40))[0]
   if not len(pos):continue
   if c['backbone']=='RW':a=np.zeros((len(pos),3));log=dict(fit=fit)
   elif c['backbone']=='MLE':a,log=matched_mle(p,fit,orig[pos])
   else:a,log=learned(p,raw if c['representation']=='RAW' else ssa,fit,orig[pos],c)
   pred[pos]=a;logs.append(dict(name=c['name'],**log))
  if not np.isfinite(pred).all():raise ValueError('incomplete forecast')
  score,rows=base.score(pred,true,years);results.append(dict(configuration=c,name=c['name'],score=score,by_year=rows));saved[c['name']]=pred
  base.dump('ALL_RESULTS.json',results);base.dump('TRAINING_LOGS.json',logs);np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved)
  print(c['name'],score,flush=True)
 agg=[]
 for rep in ['RAW','SSA']:
  for a in [.1,1.,10.]:
   rr=[r for r in results if r['configuration']['representation']==rep and r['configuration']['backbone']=='RFF' and r['configuration']['alpha']==a]
   agg.append(dict(representation=rep,alpha=a,mean_seed_loss=float(np.mean([r['score'] for r in rr]))))
 base.dump('SEED_LOSS_AGGREGATES.json',agg)
 base.dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-start,complete=True,configurations=len(results),fit_requests=len(logs),origins=len(orig),read_2022=False))
if __name__=='__main__':main()
