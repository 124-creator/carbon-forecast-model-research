from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parent
exec(compile((R/'run_synthetic.py').read_text().split('results=[]')[0],str(R/'run_synthetic.py'),'exec'))
checks=[]
for row in json.loads((R/'synthetic_results.json').read_text()):
 tag=f"{row['kind']}_{row['seed']}_{row['model']}";a=np.load(R/(tag+'_predictions.npz'),allow_pickle=False)
 mse=((a['predictions']-a['truth'])**2).mean(0)
 if not np.allclose(mse,row['mse20'],rtol=1e-12,atol=1e-12):raise RuntimeError(tag)
 logs=[json.loads(x) for x in (R/(tag+'_loss.jsonl')).read_text().splitlines()]
 if len(logs)!=300 or not all(np.isfinite(x['loss']) for x in logs):raise RuntimeError('loss log')
 checks.append(tag)
a=np.load(R/'MARKET_LOCAL_ONLY.npz',allow_pickle=False)
if a['origins'].shape!=(451,):raise RuntimeError('count')
for row in json.loads((R/'MARKET_RESULTS.json').read_text()):
 tag=f"market_{row['seed']}_{row['model']}";pred=a[tag];err=(pred-a['truth_price'])**2
 if not np.allclose(err.mean(0),row['mse20'],rtol=1e-12,atol=1e-12):raise RuntimeError(tag)
 ratio=np.mean([err[a['years']==yr,h-1].mean()/((a['truth_price']-a['rw_price'])[a['years']==yr,h-1]**2).mean() for yr in ['2020','2021'] for h in [1,5,20]])
 if not np.isclose(ratio,row['year_horizon_rw_relative_mse'],rtol=1e-12):raise RuntimeError('score')
 logs=[json.loads(x) for x in (R/(tag+'_loss.jsonl')).read_text().splitlines()]
 if len(logs)!=300 or not all(np.isfinite(x['loss']) for x in logs):raise RuntimeError('market loss')
 checks.append(tag)
# Direct native scaler check: change all future labels; shift/scale and lag normalization invariant.
m=model(NHITS,17,R/'unused.jsonl');cols=pd.Index(['y','available_mask']);x=torch.zeros(2,80,2);x[:,:,1]=1;x[:,:,0]=torch.arange(80)
def norm(v):
 w={'temporal':v.clone(),'temporal_cols':cols,'static':None,'static_cols':None};o=m._normalization(w,0);return o['temporal'][:,:60,0].clone(),m.scaler.x_shift.clone(),m.scaler.x_scale.clone()
b=norm(x);z=x.clone();z[:,60:,0]=1e6;c=norm(z)
if not all(torch.equal(u,v) for u,v in zip(b,c)):raise RuntimeError('scaler leakage')
# Native mature support count: disable sampling, preserve official padder and masks.
m.windows_batch_size=None
batch={'temporal':torch.stack([torch.arange(459,dtype=torch.float32),torch.ones(459)])[None,:,:],'temporal_cols':cols,'static':None,'static_cols':None}
w=m._create_windows(batch,'train')['temporal'];counts=w[:,60:,1].sum(0).numpy().astype(int)
if not np.array_equal(counts,np.arange(399,379,-1)):raise RuntimeError(str(counts))
rec=json.loads((R/'MARKET_RECEIPT.json').read_text());rec['protocol']=json.loads((R/'market_protocol.json').read_text());rec['native_training_windows']=int(w.shape[0]);rec['per_h_mature_labels']=counts.tolist();rec['full_h20_training_windows']=380
(R/'MARKET_RECEIPT.json').write_text(json.dumps(rec,indent=2))
summary={'passed':True,'verified_runs':len(checks),'runs':checks,'scaler_future_label_perturbation_invariant':True,'native_per_h_mature_labels':counts.tolist(),'checkpoint_loading':False,'note':'saved-score replay and direct scaler/maturity tests, not independent parent training verification'}
(R/'verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
print('SYNTHETIC NHITS',[(r['kind'],r['seed'],r['ratio']) for r in json.loads((R/'synthetic_results.json').read_text()) if r['model']=='NHITS'])
print('MARKET',[(r['model'],r['seed'],r['year_horizon_rw_relative_mse']) for r in json.loads((R/'MARKET_RESULTS.json').read_text())])
