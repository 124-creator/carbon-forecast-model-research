from pathlib import Path
import runpy,sys,json,hashlib
R=Path(__file__).resolve().parent
v=runpy.run_path(str(R/'run.py'),run_name='verify_definitions');np=v['np'];torch=v['torch'];pd=v['pd'];B=v['B']
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
p,dates,initial,origins,audit=v['load_data']();truth=np.array([p[t+1:t+21] for t in origins]);rw=np.repeat(p[origins,None],20,axis=1);years=np.array([d[:4] for d in dates[origins]])
z=np.load(R/'predictions_LOCAL_ONLY.npz',allow_pickle=False);f=np.load(B/'MARKET_LOCAL_ONLY.npz',allow_pickle=False);require(np.array_equal(z['origins'],origins),'origins');require(np.array_equal(truth,f['truth_price']),'raw truth')
receipts=json.loads((R/'fit_receipts.json').read_text());causal=json.loads((R/'causality.json').read_text());require(len(receipts)==48,'fit count');require(len(causal)==8 and all(x['parameter_identical'] and x['issuance_prediction_identical'] and x['full_fresh_fit_steps']==300 for x in causal),'actual fresh-fit causality')
cols=pd.Index(['y','available_mask']);metrics=[]
for cls in [v['NHITS'],v['DLinear']]:
 m=v['model'](cls,17,R/'unused.jsonl');m.windows_batch_size=None
 for r in [x for x in receipts if x['model']==cls.__name__]:
  log=[json.loads(line) for line in (R/(r['tag']+'_loss.jsonl')).read_text().splitlines()];require(len(log)==300 and all(np.isfinite(x['loss']) for x in log),'300 finite loss')
  y=torch.tensor(p[r['train_start']:r['train_end']+1],dtype=torch.float32);batch={'temporal':torch.stack([y,torch.ones_like(y)])[None,:,:],'temporal_cols':cols,'static':None,'static_cols':None}
  w=m._create_windows(batch,'train')['temporal'];support=w[:,60:,1].sum(0).numpy().astype(int);require(support.tolist()==r['mature_support20'],'native mature mask')
  def norm(x):
   out=m._normalization({'temporal':x.clone(),'temporal_cols':cols,'static':None,'static_cols':None},0)
   return out['temporal'][:,:60,0].clone(),m.scaler.x_shift.clone(),m.scaler.x_scale.clone()
  a=norm(w);changed=w.clone();changed[:,60:,0]+=1e6;b=norm(changed);require(all(torch.equal(x,y) for x,y in zip(a,b)),'native train scale future invariant')
 for year in ['2020','2021']:
  idx=years==year
  for h in range(1,21):
   arrays=[z[f'{cls.__name__}_{s}'] for s in [17,29]];require(all(a.shape==(451,20) and np.isfinite(a).all() for a in arrays),'complete predictions')
   mse=float(np.mean([((a[idx,h-1]-truth[idx,h-1])**2).mean() for a in arrays]));mae=float(np.mean([np.abs(a[idx,h-1]-truth[idx,h-1]).mean() for a in arrays]));base=float(((truth[idx,h-1]-rw[idx,h-1])**2).mean());frozen=float(np.mean([((f[f'market_{s}_{cls.__name__}'][idx,h-1]-truth[idx,h-1])**2).mean() for s in [17,29]]))
   metrics.append(dict(model=cls.__name__,year=year,h=h,n=int(idx.sum()),mse=mse,rmse=float(np.sqrt(mse)),mae=mae,rw_relative_mse=mse/base,frozen_relative_mse=frozen/base))
record=json.loads((R/'scores.json').read_text())
for model in ['NHITS','DLinear']:
 score=float(np.mean([r['rw_relative_mse'] for r in metrics if r['model']==model and r['h'] in [1,5,20]]));require(np.isclose(score,record['scores'][model]['score451'],rtol=1e-13),'independent score')
# Installed bytes remain exactly the executed baseline receipt, no package modification.
import neuralforecast
root=Path(neuralforecast.__file__).parent
for rel,entry in json.loads((B/'source_receipt.json').read_text())['files'].items():
 if rel!='LICENSE':require(hashlib.sha256((root/rel).read_bytes()).hexdigest()==entry['sha256'],'source identity '+rel)
result=dict(passed=True,mode='optimized' if not __debug__ else 'normal',fits_verified=48,actual_future_perturbed_fresh_fits=len(causal),scale_future_perturbation_all48=True,score_origins=451,checkpoint_loading=False,metrics=metrics)
(R/('verify_optimized.json' if not __debug__ else 'verify_normal.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:x for k,x in result.items() if k!='metrics'}));print(json.dumps([x for x in metrics if x['h'] in [1,5,20]]))
