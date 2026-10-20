"""Reproducible fixed stage-g development experiment. Never modifies stage f.

Usage: python run_experiment.py --data-dir DIR --out NEW_DIR
Input files are the exact four previously authorized archive extracts.
Per-origin arrays and sklearn binary models remain under --out/private.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, time, gzip, random, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn
from anchored_response import Response, Config, fit, DT, HORIZONS
SPECS = {
 'EUA_LONG':'142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f',
 'GUANGZHOU':'8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051',
 'HUBEI':'04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797',
 'SHANGHAI':'425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1'}
ALPHAS=(.001,.1,10.)
LRS=(.003,.01)
GAMMAS=(0.,.1,10.)
R_ALPHAS=(.0001,.001,.01,.1,1.,10.,100.)
YEARS=(2022,2023)
SEEDS=(17,29,43)

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
 if path.suffix=='.gz':
  path.write_bytes(gzip.compress(text,mtime=0))
 else: path.write_bytes(text+b'\n')
def prepare(data_dir, market, year):
 pth=Path(data_dir)/(market+'_development.csv')
 assert sha(pth)==SPECS[market], 'Source changed; do not substitute silently'
 f=pd.read_csv(pth);p=f.price.to_numpy(float);dates=pd.to_datetime(f.date).to_numpy(dtype='datetime64[D]')
 assert np.isfinite(p).all() and (p>0).all() and np.all(dates[1:]>dates[:-1])
 vol=f.volume.to_numpy(float) if 'volume' in f else np.full(len(f),np.nan)
 b=int(np.searchsorted(dates,np.datetime64(f'{year}-01-01')))
 end=int(np.searchsorted(dates,np.datetime64(f'{year+1}-01-01')))
 a=b-126
 assert a>64+20 and end-b>20
 r=np.r_[0.,np.diff(p)];scale=max(np.sqrt(np.mean(r[1:a]**2)),1e-7)
 valid=np.isfinite(vol)&(vol>=0);logs=np.zeros(len(f));logs[valid]=np.log1p(vol[valid]);vc=np.zeros((len(f),2))
 for t in range(len(f)):
  pv=valid[max(0,t-60):t]
  if valid[t] and pv.sum()>=10:
   past=logs[max(0,t-60):t][pv]
   vc[t]=[np.clip((logs[t]-np.median(past))/max(np.std(past),.25),-5,5),1]
 tokens=np.column_stack((r/scale,vc));o=np.arange(64,end-20)
 x=np.stack([tokens[t-63:t+1] for t in o]);y=np.column_stack([p[o+h]-p[o] for h in HORIZONS])
 masks={'train':o+20<a,'val':(o>=a)&(o+20<b),'final':o+20<b,'test':o>=b}
 variance=np.maximum(np.mean(y[masks['train']]**2,axis=0),scale**2*1e-6)
 meta={'market':market,'year':year,'source_sha256':SPECS[market],'source_rows':len(f),
       'a':a,'b':b,'end':end,'scale':float(scale),'variance':variance.tolist(),
       'counts':{k:int(m.sum()) for k,m in masks.items()},'volume_rows':int(valid.sum()),
       'boundaries':{k:{'first_origin':str(dates[o[m][0]]),'last_origin':str(dates[o[m][-1]]),
                        'last_label':str(dates[o[m][-1]+20])} for k,m in masks.items()},
       'max_gap_days':int(np.diff(dates).astype(int).max()),'role':'previously_exposed_development',
       'target':'archived author-reported price field, unchanged; not certified uniform close/contract'}
 return {'x':x,'y':y,'variance':variance,'scale':scale,'masks':masks,'origins':o,'dates':dates,'meta':meta}

def summary(pred,y):
 e=(pred-y[None])**2;mse=e.mean(axis=(0,1));rw=(y*y).mean(0)
 assert (rw>0).all() and np.isfinite(e).all()
 return {'Q':float(np.mean(mse/rw)), 'MSE':mse.tolist(),'RMSE':np.sqrt(mse).tolist(),
         'MAE':np.abs(pred-y[None]).mean(axis=(0,1)).tolist(),'Q_h':(mse/rw).tolist(),
         'rw_MSE':rw.tolist(),'seeds':int(pred.shape[0])}

class GRU(nn.Module):
 def __init__(self):
  super().__init__();self.rnn=nn.GRU(3,32,batch_first=True);self.head=nn.Linear(32,3)
  nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x): return self.head(self.rnn(x)[0][:,-1])
def gru_fit(x,y,v,lr,seed,mask,val=None,epochs=None):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 model=GRU();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
 xx=torch.tensor(x,dtype=torch.float32);yy=torch.tensor(y,dtype=torch.float32);vv=torch.tensor(v,dtype=torch.float32)
 ids=np.flatnonzero(mask);rng=np.random.default_rng(seed);best=float('inf');bestep=0;stale=0;steps=0;hist=[];saved=None
 cap=80 if epochs is None else epochs
 for ep in range(cap+1):
  with torch.no_grad():
   train=float(((model(xx[mask])-yy[mask])**2/vv).mean())
   vl=float(((model(xx[val])-yy[val])**2/vv).mean()) if val is not None else None
  hist.append([ep,train,vl])
  if val is not None:
   if vl<best-1e-7: best,bestep,stale,saved=vl,ep,0,{k:t.clone() for k,t in model.state_dict().items()}
   else: stale+=1
  if ep==cap or (epochs is None and ep>=15 and stale>=12):break
  for batch in np.array_split(rng.permutation(ids),max(1,int(np.ceil(len(ids)/128)))):
   opt.zero_grad(set_to_none=True);loss=((model(xx[batch])-yy[batch])**2/vv).mean();loss.backward()
   g=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(g);steps+=int(float(g)>0);opt.step()
 if epochs is None:model.load_state_dict(saved)
 receipt={'kind':'GRU32','lr':lr,'seed':seed,'selected_epoch':bestep if epochs is None else cap,'executed_epochs':hist[-1][0],
          'validation_loss':best if val is not None else None,'gradient_steps':steps,'history':hist,
          'parameters':sum(p.numel() for p in model.parameters()),'state':{k:t.detach().tolist() for k,t in model.state_dict().items()}}
 return model,receipt

def run_cell(data_dir, market, year, out):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 t0=time.perf_counter();d=prepare(data_dir,market,year);m=d['masks'];x=torch.tensor(d['x'],dtype=DT)
 y=torch.tensor(d['y']/d['scale'],dtype=DT);v=torch.tensor(d['variance']/d['scale']**2,dtype=DT)
 path=Path(out)/f'{market}_{year}';path.mkdir(parents=True,exist_ok=False)
 traces=[];models={};selections={};preds={};counter={'selection_fits':0,'final_fits':0,'hgb_estimators':0,'gradient_steps':0}
 def onefit(label, kind, cfg, *, start=None, final=False, scaled=False, epochs=None):
  subset='final' if final else 'train'
  model,rec=fit(kind,x[m[subset]],y[m[subset]],v,cfg,xv=None if final else x[m['val']],
                yv=None if final else y[m['val']],start=start,scaled=scaled,epochs=epochs)
  rec.update({'label':label,'role':'final' if final else 'selection'})
  traces.append(rec);counter['final_fits' if final else 'selection_fits']+=1;counter['gradient_steps']+=rec['gradient_steps']
  return model,rec
 def base(label,kind):
  trials=[]
  for a in ALPHAS:
   for lr in LRS:
    model,rec=onefit(label,kind,Config(a,lr));trials.append((model,rec))
  model,rec=min(trials,key=lambda t:(t[1]['validation_loss'],t[1]['config']['alpha'],t[1]['config']['lr'],t[1]['selected_epoch']))
  cfg=Config(**rec['config']);fm,fr=onefit(label,kind,cfg,final=True,epochs=rec['selected_epoch'])
  models[label]=(fm,fr);selections[label]={'config':rec['config'],'epoch':rec['selected_epoch'],'validation_loss':rec['validation_loss']}
  return model,rec,fm
 sm,srec,sfm=base('static_base','static');base('cold_duration','duration')
 base_alpha=srec['config']['alpha'];start_sel=sm.raw.detach().tolist();start_final=sfm.raw.detach().tolist()
 for label,kind in [('anchored_duration','duration'),('anchored_amplitude','amplitude'),('anchored_static','static'),('anchored_price_only','price_only')]:
  trials=[]
  for gamma in GAMMAS:
   for lr in LRS:
    model,rec=onefit(label,kind,Config(base_alpha,lr,gamma,160),start=start_sel,scaled=True);trials.append((model,rec))
  key=lambda t:(t[1]['validation_loss'],t[1]['config']['gamma'],t[1]['config']['lr'],t[1]['selected_epoch'])
  _,rec=min(trials,key=key)
  fm,fr=onefit(label,kind,Config(**rec['config']),start=start_final,final=True,scaled=True,epochs=rec['selected_epoch'])
  models[label]=(fm,fr);selections[label]={'config':rec['config'],'epoch':rec['selected_epoch'],'validation_loss':rec['validation_loss']}
  if label=='anchored_duration':
   _,zero=min([t for t in trials if t[1]['config']['gamma']==0],key=key)
   selections['warm_unpenalized']={'config':zero['config'],'epoch':zero['selected_epoch'],'validation_loss':zero['validation_loss']}
   if zero is rec:models['warm_unpenalized']=(fm,fr)
   else:
    zm,zr=onefit('warm_unpenalized','duration',Config(**zero['config']),start=start_final,final=True,scaled=True,epochs=zero['selected_epoch'])
    models['warm_unpenalized']=(zm,zr)
 with torch.no_grad():
  for name,(model,rec) in models.items():preds[name]=model(x[m['test']]).numpy()[None]*d['scale']
 obs=d['y'][m['test']];preds['rw']=np.zeros_like(obs)[None]
 feat=d['x'][:,-20:,0];target=d['y']
 scaler=StandardScaler().fit(feat[m['train']]);z=scaler.transform(feat);choices=[]
 for a in R_ALPHAS:
  model=Ridge(alpha=a*m['train'].sum()).fit(z[m['train']],target[m['train']]);vl=float(np.mean((model.predict(z[m['val']])-target[m['val']])**2/d['variance']))
  choices.append({'alpha_mean':a,'validation_loss':vl});counter['selection_fits']+=1
 sel=min(choices,key=lambda r:(r['validation_loss'],r['alpha_mean']));scaler=StandardScaler().fit(feat[m['final']]);z=scaler.transform(feat)
 model=Ridge(alpha=sel['alpha_mean']*m['final'].sum()).fit(z[m['final']],target[m['final']]);counter['final_fits']+=1
 preds['ridge_price']=model.predict(z[m['test']])[None];selections['ridge_price']={'selected':sel,'trials':choices,'coefficients':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()}
 feat=np.column_stack((feat,d['x'][:,-1,1:]))
 choices=[]
 for leaves in (7,15):
  for l2 in (.1,1.):
   est=[HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_leaf_nodes=leaves,l2_regularization=l2,early_stopping=False,random_state=17).fit(feat[m['train']],target[m['train'],j]) for j in range(3)]
   pp=np.column_stack([e.predict(feat[m['val']]) for e in est]);vl=float(np.mean((pp-target[m['val']])**2/d['variance']))
   choices.append({'leaves':leaves,'l2':l2,'validation_loss':vl});counter['selection_fits']+=1;counter['hgb_estimators']+=3
 sel=min(choices,key=lambda r:(r['validation_loss'],r['leaves'],r['l2']))
 est=[HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_leaf_nodes=sel['leaves'],l2_regularization=sel['l2'],early_stopping=False,random_state=17).fit(feat[m['final']],target[m['final'],j]) for j in range(3)]
 counter['final_fits']+=1;counter['hgb_estimators']+=3
 preds['hgb']=np.column_stack([e.predict(feat[m['test']]) for e in est])[None];selections['hgb']={'selected':sel,'trials':choices}
 private=path/'private';private.mkdir();(private/'hgb.pkl').write_bytes(pickle.dumps(est))
 choices=[]
 for lr in LRS:
  gm,gr=gru_fit(d['x'],y.numpy(),v.numpy(),lr,17,m['train'],m['val']);gr.update({'label':'gru','role':'selection'});traces.append(gr)
  counter['selection_fits']+=1;counter['gradient_steps']+=gr['gradient_steps'];choices.append(gr)
 sel=min(choices,key=lambda r:(r['validation_loss'],r['lr'],r['selected_epoch']));gpred=[]
 for seed in SEEDS:
  gm,gr=gru_fit(d['x'],y.numpy(),v.numpy(),sel['lr'],seed,m['final'],epochs=sel['selected_epoch']);gr.update({'label':'gru','role':'final'});traces.append(gr)
  counter['final_fits']+=1;counter['gradient_steps']+=gr['gradient_steps']
  with torch.no_grad():gpred.append(gm(torch.tensor(d['x'][m['test']],dtype=torch.float32)).numpy()*d['scale'])
 preds['gru']=np.asarray(gpred);selections['gru']={'lr':sel['lr'],'epoch':sel['selected_epoch'],'validation_loss':sel['validation_loss']}
 result={name:summary(pred,obs) for name,pred in preds.items()}
 np.savez_compressed(private/'predictions.npz',observed=obs,origins=d['origins'][m['test']],**preds)
 dump(path/'RESULTS.json',{'meta':d['meta'],'scores':result,'selection':selections,'counter':counter,'wall_seconds':time.perf_counter()-t0})
 dump(path/'TRACES.json.gz',traces)
 return {'cell':f'{market}_{year}','result':result,'counter':counter,'wall_seconds':time.perf_counter()-t0}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
 torch.set_num_threads(1)
 out=Path(args.out)
 if out.exists() and any(out.iterdir()):raise FileExistsError('New output directory required')
 out.mkdir(parents=True,exist_ok=True)
 cells=[(k,y) for k in SPECS for y in YEARS];manifests=[prepare(args.data_dir,k,y)['meta'] for k,y in cells]
 code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
 protocol={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_sha256':code,'data':manifests,'alphas':ALPHAS,'learning_rates':LRS,'gammas':GAMMAS,'horizons':HORIZONS,'years':YEARS,'seeds_gru':SEEDS,'primary':'anchored_duration','all_development':True,'warm_start_from_training_only_static_fit':True,'normalization':'initial train price RMS and horizon target moments; context RMS recomputed only on fitting subset','runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__},'preflight_only':args.preflight}
 dump(out/'PROTOCOL.json',protocol)
 if args.preflight:
  print(json.dumps({'status':'PREFLIGHT_NO_FITS','cells':manifests},ensure_ascii=False));return
 ans=[]
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futs={pool.submit(run_cell,args.data_dir,k,y,args.out):(k,y) for k,y in cells}
  for f in as_completed(futs):
   a=f.result();ans.append(a);print(json.dumps({'completed':a['cell'],'Q':{k:v['Q'] for k,v in a['result'].items()}},ensure_ascii=False),flush=True)
 assert code=={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
 dump(out/'COMPLETE.json',{'status':'COMPLETE','cells':len(ans),'runs':sorted(ans,key=lambda a:a['cell']),'source_unchanged':True})
if __name__=='__main__':main()
