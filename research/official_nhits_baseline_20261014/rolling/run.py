"""Fixed refresh-only official baseline; no weight loading or package edits."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
R=Path(__file__).resolve().parent; B=R.parent
# Execute definitions only; do NOT execute receipt() or synthetic experiments.
src=(B/'run_synthetic.py').read_text();exec(compile(src.split('receipt()\nresults=[]')[0].split('def receipt():')[0],str(B/'run_synthetic.py'),'exec'))
R=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parent/'torch_segmented_backbone_20261014'))
from prefix_loader import read_prefix
class BudgetExpired(Exception):pass
budget_seconds=7.156000000075437;fit_start=None # includes preserved first diagnostic attempt
class Budget(Callback):
 def on_train_batch_end(self,*args,**kwargs):
  if budget_seconds+time.monotonic()-fit_start>=600:raise BudgetExpired('600 second actual fit budget exhausted')
def savej(name,obj):
 p=R/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');tmp.replace(p)
def load_data():
 rows,audit=read_prefix(B/'../../data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
 p=np.array([float(r['author_reported_price']) for r in rows]);dates=np.array([r['date'] for r in rows]);initial=int(np.sum(dates<'2020-01-01'))
 origins=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 if len(origins)!=451:raise RuntimeError('451 required')
 return p,dates,initial,origins,audit
def fingerprint(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def fit_predict(p,t,ts,cls,seed,tag):
 global fit_start,budget_seconds
 # 480 origins with at least one mature label, each with 60 full lags.
 # Official native right padding retains horizon-specific maturity mask.
 start=max(0,t+1-540);train=p[start:t+1].copy()
 log=R/(tag+'_loss.jsonl');m=model(cls,seed,log);m.trainer_kwargs['callbacks'].append(Budget())
 nf=NeuralForecast(models=[m],freq=1);fit_start=time.monotonic()
 try:nf.fit(df=frame(train))
 finally:budget_seconds+=time.monotonic()-fit_start
 infer=pd.concat([pd.DataFrame({'unique_id':f'o{x:04d}','ds':np.arange(60),'y':p[x-59:x+1]}) for x in ts],ignore_index=True)
 out=nf.predict(df=infer).reset_index();pred=np.array([out[out.unique_id==f'o{x:04d}'].sort_values('ds')[cls.__name__].to_numpy() for x in ts])
 if pred.shape!=(len(ts),20) or not np.isfinite(pred).all():raise RuntimeError('invalid prediction')
 count=len(train)-60
 return pred,dict(tag=tag,t=int(t),seed=seed,model=cls.__name__,train_start=start,train_end=int(t),available_windows=count,mature_support20=[max(0,count-h+1) for h in range(1,21)],steps=len(log.read_text().splitlines()),parameter_sha256=fingerprint(m),fit_seconds_budget_total=budget_seconds)
def replay():
 z=np.load(R/'predictions_LOCAL_ONLY.npz',allow_pickle=False);f=np.load(B/'MARKET_LOCAL_ONLY.npz',allow_pickle=False)
 origins=z['origins'];truth=f['truth_price'];rw=f['rw_price'];years=f['years'];rows=[];scores={}
 if not np.array_equal(origins,f['origins']):raise RuntimeError('origin identity')
 for cls in ['NHITS','DLinear']:
  arrays=[z[f'{cls}_{s}'] for s in [17,29]];valid=np.logical_and.reduce([np.isfinite(a).all(axis=1) for a in arrays]);full=bool(valid.all()); vals=[]
  for year in ['2020','2021']:
   idx=(years==year)&valid
   for h in range(1,21):
    if not idx.any():continue
    mse=float(np.mean([((a[idx,h-1]-truth[idx,h-1])**2).mean() for a in arrays]));base=float(((rw[idx,h-1]-truth[idx,h-1])**2).mean())
    frozen=float(np.mean([((f[f'market_{s}_{cls}'][idx,h-1]-truth[idx,h-1])**2).mean() for s in [17,29]]))
    rows.append(dict(model=cls,year=year,h=h,n=int(idx.sum()),refresh_mse=mse,frozen_same_origins_mse=frozen,rw_mse=base,refresh_rw_ratio=mse/base,frozen_rw_ratio=frozen/base))
    if h in [1,5,20]:vals.append(mse/base)
  scores[cls]=dict(full_451=full,complete_both_seed_origins=int(valid.sum()),score451=float(np.mean(vals)) if full else None,partial_score_not_451=float(np.mean(vals)) if vals and not full else None)
 savej('scores.json',dict(scores=scores,year_h20=rows,seed_aggregation='mean losses, not mean predictions',historical_strong='MLE and strong linear are historical reused comparators, NOT retrained in this refresh experiment',candidate_endorsement=False))
 print(json.dumps(scores),flush=True)
def run():
 global budget_seconds
 if (R/'predictions_LOCAL_ONLY.npz').exists():raise RuntimeError('fresh run only; existing artifacts cannot be overwritten')
 p,dates,initial,origins,audit=load_data();blocks=[(t,np.array([x for x in origins if t<=x<t+40])) for t in range(initial,int(origins[-1])+1,40)]
 savej('protocol.json',dict(refresh_observations=40,recent_available_windows=480,native_partial_horizon_mask=True,h=20,input_size=60,max_steps=300,seeds=[17,29],models=['NHITS','DLinear'],budget_fit_seconds=600,only_change='refresh and recent-window training support; exact original architecture optimizer loss scaler settings',no_pretraining=True,no_checkpoint_loading=True,audit=audit,initial=initial,blocks=[int(t) for t,_ in blocks],baseline=str(B)))
 saved={'origins':origins};receipts=[];causal=[]
 for cls in [NHITS,DLinear]:
  for seed in [17,29]:saved[f'{cls.__name__}_{seed}']=np.full((451,20),np.nan)
 def persist():
  np.savez(R/'predictions_LOCAL_ONLY.npz',**saved);savej('fit_receipts.json',receipts);savej('causality.json',causal);savej('budget.json',dict(actual_fit_seconds=budget_seconds,status='running'))
 persist()
 try:
  # Run block-first so timeout retains fair paired seed/model coverage.
  for block,(t,ts) in enumerate(blocks):
   for cls in [NHITS,DLinear]:
    for seed in [17,29]:
     tag=f'b{block:02d}_{cls.__name__}_{seed}'
     pred,receipt=fit_predict(p,t,ts,cls,seed,tag);saved[f'{cls.__name__}_{seed}'][np.searchsorted(origins,ts)]=pred;receipts.append(receipt);persist()
     if block==0:
      changed=p.copy();changed[t+1:]+=10000+np.arange(len(p)-t-1)*17
      repeat,rr=fit_predict(changed,t,ts,cls,seed,tag+'_future_changed')
      check=dict(model=cls.__name__,seed=seed,t=int(t),full_fresh_fit_steps=rr['steps'],parameter_identical=rr['parameter_sha256']==receipt['parameter_sha256'],issuance_prediction_identical=bool(np.array_equal(repeat[0],pred[0])),perturb_from=int(t+1),repeat_receipt=rr)
      causal.append(check);persist()
      if not check['parameter_identical'] or not check['issuance_prediction_identical']:raise RuntimeError('causality failure')
  status='complete'
 except BudgetExpired as e:status='timeout';print(str(e),flush=True)
 except Exception as e:
  savej('budget.json',dict(actual_fit_seconds=budget_seconds,status='error',error=repr(e)));raise
 persist();savej('budget.json',dict(actual_fit_seconds=budget_seconds,status=status,completed_fits=len(receipts),no_missing_predictions_filled=True));replay()
if __name__=='__main__':
 if '--replay' in sys.argv:replay()
 else:run()
