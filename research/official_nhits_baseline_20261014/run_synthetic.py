"""Official installed NeuralForecast 1.7.7; no surrogate architecture, no pretraining."""
from pathlib import Path
import os,json,hashlib,inspect,importlib.metadata as md,subprocess,time
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np,pandas as pd,torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS,MLP,DLinear
from neuralforecast.losses.pytorch import MSE
from pytorch_lightning.callbacks import Callback
R=Path(__file__).resolve().parent
torch.set_num_threads(1)
class LossTrace(Callback):
 def __init__(self,path): self.path=path
 def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
  loss=outputs['loss'] if isinstance(outputs,dict) else outputs
  with self.path.open('a') as f: f.write(json.dumps({'step':trainer.global_step,'loss':float(loss.detach()),'epoch':trainer.current_epoch})+'\n')
def frame(y): return pd.DataFrame({'unique_id':'s','ds':np.arange(len(y)),'y':y})
def model(cls,seed,path):
 kw=dict(h=20,input_size=60,loss=MSE(),max_steps=300,learning_rate=0.001,num_lr_decays=-1,windows_batch_size=256,scaler_type='standard',random_seed=seed,accelerator='cpu',devices=1,enable_checkpointing=False,enable_progress_bar=False,logger=False,enable_model_summary=False,callbacks=[LossTrace(path)],val_check_steps=100)
 if cls==NHITS: kw.update(mlp_units=[[64,64]]*3,n_blocks=[1,1,1],n_pool_kernel_size=[4,2,1],n_freq_downsample=[4,2,1])
 if cls==MLP: kw.update(hidden_size=64,num_layers=2)
 return cls(**kw)
def generate(kind):
 rng=np.random.default_rng(817); n=1000;t=np.arange(n)
 if kind=='trend': return .03*t+np.sin(t/18)*.8+rng.normal(0,.03,n)
 if kind=='transient':
  state=0.;out=[]
  for k in t:
   state=.85*state+(4. if k%35==0 else 0.)
   out.append(state)
  return np.array(out)+rng.normal(0,.03,n)
 return rng.normal(0,1,n)
def receipt():
 files={}; dst=R/'official_source';dst.mkdir(exist_ok=True)
 import neuralforecast
 root=Path(neuralforecast.__file__).parent
 for rel in ['models/nhits.py','models/mlp.py','models/dlinear.py','common/_base_windows.py','common/_base_model.py','common/_scalers.py','core.py','tsdataset.py','losses/pytorch.py']:
  src=root/rel
  if src.exists():
   b=src.read_bytes(); (dst/rel.replace('/','__')).write_bytes(b);files[rel]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 dist=md.distribution('neuralforecast')
 for f in dist.files:
  if 'LICENSE' in str(f).upper():
   b=dist.locate_file(f).read_bytes(); (dst/'LICENSE').write_bytes(b); files['LICENSE']={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 versions={d.metadata['Name']:d.version for d in md.distributions()}
 (R/'dependency_lock.json').write_text(json.dumps(versions,indent=2))
 (R/'requirements.lock.txt').write_text('\n'.join(k+'=='+v for k,v in sorted(versions.items()))+'\n')
 (R/'source_receipt.json').write_text(json.dumps({'version':md.version('neuralforecast'),'torch':torch.__version__,'files':files,'pretraining':False,'checkpoint_loading':False,'source':'installed official PyPI wheel, not previously reviewed floating main; hashes fix executed files','cpu_libraries':'parent CPU package bytes hardlinked read-only; independent project venv; no Herm es package modification'},indent=2))
receipt()
results=[]
for kind in ['trend','transient','no_signal']:
 y=generate(kind);np.save(R/(kind+'_series.npy'),y)
 for seed in [17,29]:
  for cls in [NHITS,MLP,DLinear]:
   tag=f'{kind}_{seed}_{cls.__name__}'; log=R/(tag+'_loss.jsonl')
   if log.exists():log.unlink()
   nf=NeuralForecast(models=[model(cls,seed,log)],freq=1)
   started=time.time(); nf.fit(df=frame(y[:700]))
   preds=[];origins=np.arange(700,980,20)
   for end in origins:
    out=nf.predict(df=frame(y[:end]))
    preds.append(out[cls.__name__].to_numpy())
   preds=np.array(preds);truth=np.array([y[e:e+20] for e in origins]);rw=np.array([np.repeat(y[e-1],20) for e in origins])
   delta_truth=truth-y[origins-1,None];delta_pred=preds-y[origins-1,None]
   loss=((preds-truth)**2).mean(axis=0);base=((rw-truth)**2).mean(axis=0)
   np.savez(R/(tag+'_predictions.npz'),origins=origins,predictions=preds,truth=truth,rw=rw,price_mse=loss,change_mse=((delta_pred-delta_truth)**2).mean(axis=0))
   # Never load these checkpoints: evidentiary snapshots only, initial prefix fit.
   torch.save(nf.models[0].state_dict(),R/(tag+'_trained_state.pt'))
   row={'kind':kind,'seed':seed,'model':cls.__name__,'mse20':loss.tolist(),'rw_mse20':base.tolist(),'ratio':float(loss.mean()/base.mean()),'seconds':time.time()-started,'steps':len(log.read_text().splitlines())}
   results.append(row);(R/'synthetic_results.json').write_text(json.dumps(results,indent=2));print(row,flush=True)
nh=[r for r in results if r['model']=='NHITS']
gate=all(r['ratio']<.5 for r in nh if r['kind']!='no_signal') and all(r['ratio']<=1.1 for r in nh if r['kind']=='no_signal')
(R/'gate.json').write_text(json.dumps({'passed':gate,'criteria':'each NHITS seed: trend/transient pooled h1..20 MSE <0.5 RW; no signal <=1.1 RW. This is a learnability check, not market advantage.','market_authorized_by_this_run':gate,'market_executed':False},indent=2))
print('GATE',gate,flush=True)
