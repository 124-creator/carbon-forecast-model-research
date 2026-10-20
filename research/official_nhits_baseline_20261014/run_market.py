from pathlib import Path
import sys,json,time
R=Path(__file__).resolve().parent
# Reuse definitions only, not synthetic training loop; exact source is receipted.
exec(compile((R/'run_synthetic.py').read_text().split('results=[]')[0],str(R/'run_synthetic.py'),'exec'))
sys.path.insert(0,str(R.parent/'torch_segmented_backbone_20261014'))
from prefix_loader import read_prefix
if not json.loads((R/'gate.json').read_text())['passed']:raise RuntimeError('synthetic gate failed')
rows,audit=read_prefix(R/'../../data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
p=np.array([float(r['author_reported_price']) for r in rows]); dates=np.array([r['date'] for r in rows]);initial=int(np.sum(dates<'2020-01-01'))
origins=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
if len(origins)!=451:raise RuntimeError('origin count mismatch')
truth=np.array([p[t+1:t+21] for t in origins]);rw=np.repeat(p[origins,None],20,axis=1);years=np.array([d[:4] for d in dates[origins]])
# Each inference series contains only 60 issuance-available observations.
infer=pd.concat([pd.DataFrame({'unique_id':f'o{t:04d}','ds':np.arange(60),'y':p[t-59:t+1]}) for t in origins],ignore_index=True)
saved=dict(origins=origins,truth_price=truth,truth_change=truth-p[origins,None],rw_price=rw,years=years,issuance_price=p[origins]);results=[]
for seed in [17,29]:
 for cls in [NHITS,MLP,DLinear]:
  tag=f'market_{seed}_{cls.__name__}';log=R/(tag+'_loss.jsonl');started=time.time()
  if log.exists():log.unlink()
  nf=NeuralForecast(models=[model(cls,seed,log)],freq=1);nf.fit(df=frame(p[:initial]))
  out=nf.predict(df=infer).reset_index();pred=np.array([out[out.unique_id==f'o{t:04d}'].sort_values('ds')[cls.__name__].to_numpy() for t in origins])
  if pred.shape!=(451,20) or not np.isfinite(pred).all():raise RuntimeError('prediction shape/finiteness')
  saved[tag]=pred
  err=(pred-truth)**2;errchange=(pred-p[origins,None]-(truth-p[origins,None]))**2
  if not np.allclose(err,errchange,atol=1e-10):raise RuntimeError('price/change identity')
  mse=err.mean(axis=0);rmse=np.sqrt(mse);mae=np.abs(pred-truth).mean(axis=0)
  score=float(np.mean([err[years==yr,h-1].mean()/((truth-rw)[years==yr,h-1]**2).mean() for yr in ['2020','2021'] for h in [1,5,20]]))
  row=dict(model=cls.__name__,seed=seed,mse20=mse.tolist(),rmse20=rmse.tolist(),mae20=mae.tolist(),selected_h=[1,5,20],year_horizon_rw_relative_mse=score,seconds=time.time()-started,steps=len(log.read_text().splitlines()))
  results.append(row);np.savez(R/'MARKET_LOCAL_ONLY.npz',**saved);(R/'MARKET_RESULTS.json').write_text(json.dumps(results,indent=2));torch.save(nf.models[0].state_dict(),R/(tag+'_trained_state.pt'));print(row,flush=True)
(R/'MARKET_RECEIPT.json').write_text(json.dumps(dict(audit=audit,initial=initial,origins=451,training_max_index=initial-1,training_max_date=str(dates[initial-1]),full_h20_training_windows=initial-60-20+1,protocol=json.loads((R/'market_protocol.json').read_text()),pretraining=False,loaded_checkpoints=False,market_data_local_only=True),indent=2))
