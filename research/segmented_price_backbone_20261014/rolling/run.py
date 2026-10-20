"""Fixed-config rolling refresh diagnostic; no tuning, no boundary read."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib
from pathlib import Path
import numpy as np
from autograd import grad
from autograd.misc.flatten import flatten
OUT=Path(__file__).resolve().parent; BASE=OUT.parent; ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
import model
from run_market import read_prefix
H=model.H
START=time.monotonic(); DEADLINE=START+850

def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def dataset(p,end):
 idx=np.arange(max(59,end-20-480),end-1);x=np.stack([p[t-59:t+1] for t in idx]);y=np.full((len(idx),3),np.nan)
 for j,h in enumerate(H):
  ok=(idx+h<end)&(idx>=end-h-480);y[ok,j]=p[idx[ok]+h]-p[idx[ok]]
 sx=max(np.std(x-x[:,-1:]),.05);sy=np.maximum(np.nanstd(y,axis=0),.05)
 return x/sx,y/sy,sx,sy,dict(end_exclusive=end,n_windows=len(idx),per_h_counts=np.isfinite(y).sum(0).tolist(),max_target=[int(np.max(idx[np.isfinite(y[:,j])]+h)) for j,h in enumerate(H)],input_scale=float(sx),target_scale=sy.tolist())
def train(x,y,seed):
 p=model.initialize(seed);f,u=flatten(p);m=np.zeros_like(f);v=m.copy();rng=np.random.RandomState(seed);g=grad(lambda ff,xx,yy:model.loss(u(ff),xx,yy,'segmented'));history=[];norms=[];k=0;initial=float(model.loss(p,x,y,'segmented'))
 for ep in range(60):
  for ids in np.array_split(rng.permutation(len(x)),8):
   if time.monotonic()>DEADLINE:raise TimeoutError('bounded 850s compute budget')
   df=g(f,x[ids],y[ids]);norm=float(np.linalg.norm(df));norms.append(norm);df=df/max(1,norm)+1e-4*f;k+=1;m=.9*m+.1*df;v=.999*v+.001*df**2;f-=.001*(m/(1-.9**k))/(np.sqrt(v/(1-.999**k))+1e-8)
  history.append(float(model.loss(u(f),x,y,'segmented')))
 fullnorm=float(np.linalg.norm(g(f,x,y)))
 return u(f),dict(initial=initial,history=history,gradient_norms=norms,final_full_gradient_norm=fullnorm,updates=k,effective_nonzero_gradient_steps=int(np.count_nonzero(np.array(norms)>1e-12)),parameters=len(f),last10_loss_drop=history[-11]-history[-1],adequacy='fixed budget fully executed; finite gradients/loss descent do not certify optimum')
def score(pred,truth,years):
 cells=[]
 for year in ['2020','2021']:
  ok=years==year;cells.append(dict(year=year,n=int(ok.sum()),mse_h=np.mean((pred[ok]-truth[ok])**2,0).tolist(),relative_mse_h=(np.mean((pred[ok]-truth[ok])**2,0)/np.mean(truth[ok]**2,0)).tolist()))
 return float(np.mean([c['relative_mse_h'] for c in cells])),cells

def main():
 if (OUT/'RESULTS.json').exists():raise RuntimeError('preserve prior outputs')
 dates,p,audit=read_prefix();end=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(end,930) if dates[t][:4]==dates[t+20][:4]]);years=np.array([s[:4] for s in dates[orig]]);truth=p[orig[:,None]+H]-p[orig,None]
 if len(orig)!=451 or end!=459:raise ValueError('protocol')
 saved=dict(origins=orig,years=years,truth=truth);rows=[];logs=[];checks=[]
 def retain(name,pred,source):
  s,c=score(pred,truth,years);saved[name]=pred;rows.append(dict(name=name,score=s,cells=c,source=source));np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved);dump('RESULTS.json',rows);print(name,s,flush=True)
 # Re-score preserved frozen artifacts against independently constructed labels.
 for mode in ['linear','segmented','unsegmented']:
  for seed in [7,17]:
   z=np.load(BASE/f'market_{seed}_{mode}.npz',allow_pickle=False)
   if not np.array_equal(z['origins'],orig) or not np.array_equal(z['target'],truth):raise ValueError('frozen labels')
   retain(f'frozen_{mode}_{seed}',z['prediction'],'historical frozen saved forecast reuse')
 # Strong historical issuance-vintage forecasts, identical origin/label verification.
 z=np.load(ROOT/'research/coverage_interface_audit_20261014/PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
 if not np.array_equal(z['origins'],orig) or not np.array_equal(z['truth'],truth):raise ValueError('historical support')
 retain('historical_state_linear_a10',z['historical_native'],'historical reuse; mature residual calibration; NOT raw segment linear')
 z=np.load(ROOT/'research/multi_angle_model_search_20261013/state/PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
 if not np.array_equal(z['origins'],orig):raise ValueError('MLE support')
 retain('historical_MLE480',z['issuance_mle'][orig],'historical reuse; rolling40 MLE; one-step likelihood support differs')
 xx,yy,sx,sy,support=dataset(p,end);pert=p.copy();pert[end:]+=1000+np.arange(len(p)-end)*17;px,py,psx,psy,_=dataset(pert,end)
 if not np.array_equal(xx,px) or not np.array_equal(yy,py,equal_nan=True) or sx!=psx or not np.array_equal(sy,psy):raise ValueError('future training inputs')
 initial={}
 for seed in [7,17]:
  a,la=train(xx,yy,seed);b,lb=train(px,py,seed);fa,_=flatten(a);fb,_=flatten(b);diff=float(np.max(np.abs(fa-fb)));pred=model.predict(a,np.stack([p[t-59:t+1] for t in orig])/sx,'segmented')*sy;zd=np.load(BASE/f'market_{seed}_segmented.npz',allow_pickle=False);replay=float(np.max(np.abs(pred-zd['prediction'])));fd=float(np.max(np.abs(model.predict(a,p[399:459][None]/sx,'segmented')-model.predict(b,pert[399:459][None]/psx,'segmented'))));check=dict(seed=seed,suffix_start=end,perturb_all_future=True,parameter_maxdiff=diff,frozen_retraining_prediction_maxdiff=replay,last_prefix_forecast_maxdiff=fd)
  checks.append(check);dump('INITIAL_TRAINING_INVARIANCE.json',checks)
  if max(diff,replay,fd)>1e-10:raise ValueError('initial training invariance/replay')
  initial[seed]=(a,la);logs.append(dict(policy='initial_replay',seed=seed,**support,**la));dump('TRAINING_LOGS.json',logs)
 rolling={seed:np.full(truth.shape,np.nan) for seed in [7,17]};linear=np.full(truth.shape,np.nan)
 for fit in range(end,len(p),40):
  pos=np.where((orig>=fit)&(orig<fit+40))[0]
  if not len(pos):continue
  x,y,sx,sy,support=dataset(p,fit);test=np.stack([p[t-59:t+1] for t in orig[pos]])/sx
  X=np.column_stack([np.ones(len(x)),x-x[:,-1:]]);T=np.column_stack([np.ones(len(test)),test-test[:,-1:]]);pen=np.eye(61);pen[0,0]=0
  for j in range(3):
   ok=np.isfinite(y[:,j]);coef=np.linalg.solve(X[ok].T@X[ok]+pen,X[ok].T@y[ok,j]);linear[pos,j]=(T@coef)*sy[j]
  logs.append(dict(policy='rolling_linear',**support,alpha=1.,parameters=183,solver='exact ridge'))
  for seed in [7,17]:
   a,log=initial[seed] if fit==end else train(x,y,seed);rolling[seed][pos]=model.predict(a,test,'segmented')*sy;logs.append(dict(policy='rolling_segmented',seed=seed,**support,**log));dump('TRAINING_LOGS.json',logs);np.savez_compressed(OUT/f'checkpoint_{seed}.npz',origins=orig,prediction=rolling[seed],truth=truth,years=years);print('fit',fit,'seed',seed,'loss',log['initial'],log['history'][-1],flush=True)
 dump('AUDIT.json',dict(**audit,seconds=time.monotonic()-START,budget_seconds=900,seeds=[7,17],refresh=40,max_per_h_mature=480,fit_policy='fresh fixed seed initialization each refresh; 60epochs/480updates; initial same as preserved frozen',only_new_fair_factor='rolling parameter refresh with fixed latest480 mature support and train-only scales',development_only=True,not_innovation=True,source_model_sha256=hashlib.sha256((BASE/'model.py').read_bytes()).hexdigest()))
 retain('rolling_linear',linear,'executed fresh rolling40 raw segment ridge alpha1')
 for seed in [7,17]:
  if not np.isfinite(rolling[seed]).all():raise ValueError('incomplete')
  retain(f'rolling_segmented_{seed}',rolling[seed],'executed rolling40 fixed-config SegRNN')
 dump('COMPLETED.json',dict(seconds=time.monotonic()-START,models=len(rows),origins=len(orig)))
if __name__=='__main__':main()
