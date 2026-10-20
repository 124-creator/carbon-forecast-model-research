"""Read-only numerical replay and the predeclared exploratory block analysis.

No nonlinear model training. Structural basis is recomputed by a NumPy
recurrence, readouts by augmented SVD least squares. HGB/GRU forward replays
are verification of stored predictors, not independently reimplemented models.
"""
from __future__ import annotations
import argparse,gzip,json,pickle,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
import run_experiment as run
from test_model import numpy_basis

PRIMARY='anchored_duration'
COMP=['cold_duration','static_base','anchored_static','anchored_amplitude','anchored_price_only','ridge_price','rw','warm_unpenalized']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stable_errors(p,y):
 e=(p-y[None])**2
 return e[0]+(e-e[0]).mean(0)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--data-dir',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
 torch.set_num_threads(1)
 inp=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 complete=json.load(open(inp/'COMPLETE.json'));assert complete['status']=='COMPLETE' and complete['cells']==8
 protocol=json.load(open(inp/'PROTOCOL.json'))
 assert protocol['source_sha256']=={p.name:digest(p) for p in (ROOT/'source').glob('*.py')}
 allrows=[];configs=[];states={};arrays=[];counters=[];identities=[]
 gaps={'basis':0.,'head_svd':0.,'validation_loss':0.,'final_prediction':0.,'reported_Q':0.,'gru_forward':0.,'hgb_forward':0.}
 replay_count=0;gru_count=0;gate_selected=[];fit_positive=0;linear_solves=0;zero_final=0
 cells=sorted(p for p in inp.iterdir() if p.is_dir())
 for cell in cells:
  result=json.load(open(cell/'RESULTS.json'));meta=result['meta'];market,year=meta['market'],meta['year'];d=run.prepare(args.data_dir,market,year);m=d['masks'];x=d['x'];y=d['y']/d['scale'];v=d['variance']/d['scale']**2
  npz=np.load(cell/'private/predictions.npz');target=npz['observed'];origins=npz['origins']
  f=pd.read_csv(Path(args.data_dir)/(market+'_development.csv'));p=f.price.to_numpy(float)
  assert np.array_equal(target,np.column_stack([p[origins+h]-p[origins] for h in (1,5,20)]))
  assert np.array_equal(origins,d['origins'][m['test']])
  assert d['origins'][m['train']][-1]+20<meta['a'] and d['origins'][m['val']][-1]+20<meta['b']
  assert origins[-1]+20<meta['end']
  for j,t in enumerate(d['origins']):
   assert np.array_equal(x[j,:,0],np.diff(p[t-64:t+1])/d['scale'])
  traces=json.loads(gzip.decompress((cell/'TRACES.json.gz').read_bytes()))
  final_by_label={};states[cell.name]={};identities.append({'cell':cell.name,'predictions_sha256':digest(cell/'private/predictions.npz'),'full_trace_sha256':digest(cell/'TRACES.json.gz'),'source_sha256':run.SPECS[market]})
  assert len(traces)==result['counter']['selection_fits']+result['counter']['final_fits']-13
  for r in traces:
   fit_positive+=int(r['gradient_steps']>0)
   zero_final+=int(r['role']=='final' and r['selected_epoch']==0)
   hist=r['history'];c=r.get('config',{});ix=3 if 'model' in r else 2
   configs.append({'cell':cell.name,'label':r['label'],'role':r['role'],'alpha':c.get('alpha'),'gamma':c.get('gamma'),'lr':c.get('lr',r.get('lr')),'seed':r.get('seed'),'selected_epoch':r['selected_epoch'],'executed_epochs':r['executed_epochs'],'gradient_steps':r['gradient_steps'],'validation_selected':r['validation_loss'],'initial_training_loss':hist[0][1],'last_training_loss':hist[-1][1],'initial_validation_loss':hist[0][ix],'last_validation_loss':hist[-1][ix]})
   mask=m['final'] if r['role']=='final' else m['train']
   if 'model' in r:
    saved=r['model'];phi=numpy_basis(x[mask],saved);beta=np.asarray(saved['state']['beta']);aa=(phi/np.sqrt(v)[None,:,None]).reshape(-1,3);bb=(y[mask]/np.sqrt(v)).reshape(-1);s=np.maximum(np.sqrt((aa*aa).mean(0)),1e-8)
    betasvd=np.linalg.lstsq(np.vstack((aa/s/np.sqrt(len(aa)),np.sqrt(c['alpha'])*np.eye(3))),np.r_[bb/np.sqrt(len(aa)),np.zeros(3)],rcond=None)[0]/s
    gaps['head_svd']=max(gaps['head_svd'],float(np.max(abs(beta-betasvd))))
    tm=run.Response.restore(saved)
    with torch.no_grad():gaps['basis']=max(gaps['basis'],float(np.max(abs(phi-tm.basis(torch.tensor(x[mask],dtype=run.DT)).numpy()))))
    if r['role']=='selection':
     vp=numpy_basis(x[m['val']],saved)@beta;vl=float(np.mean((vp-y[m['val']])**2/v));gaps['validation_loss']=max(gaps['validation_loss'],abs(vl-r['validation_loss']))
     best=float('inf');bestep=0
     for row in hist:
      if row[3]<best-1e-7:best,bestep=row[3],row[0]
     assert bestep==r['selected_epoch']
    else:
     final_by_label[r['label']]=r
     pred=(numpy_basis(x[m['test']],saved)@beta)*d['scale'];gaps['final_prediction']=max(gaps['final_prediction'],float(np.max(abs(pred-npz[r['label']][0]))))
     states[cell.name][r['label']]={'model':saved,'config':c,'selected_epoch':r['selected_epoch'],'anchor_rates_raw':r['anchor_rates_raw']}
    replay_count+=1;linear_solves+=len(hist)
   else:
    gm=run.GRU();gm.load_state_dict({k:torch.tensor(z,dtype=torch.float32) for k,z in r['state'].items()})
    if r['role']=='final':
     with torch.no_grad():pp=gm(torch.tensor(x[m['test']],dtype=torch.float32)).numpy()*d['scale']
     si=run.SEEDS.index(r['seed']);gaps['gru_forward']=max(gaps['gru_forward'],float(np.max(abs(pp-npz['gru'][si]))));gru_count+=1
  if 'warm_unpenalized' not in final_by_label:
   assert result['selection']['warm_unpenalized']==result['selection']['anchored_duration']
   assert np.array_equal(npz['warm_unpenalized'],npz['anchored_duration'])
   states[cell.name]['warm_unpenalized']={'alias':'anchored_duration'}
  for label in ('static_base','cold_duration','anchored_duration','anchored_static','anchored_amplitude','anchored_price_only'):
   trials=[r for r in traces if r['label']==label and r['role']=='selection']
   primary_key='alpha' if label in ('static_base','cold_duration') else 'gamma'
   chosen=min(trials,key=lambda r:(r['validation_loss'],r['config'][primary_key],r['config']['lr'],r['selected_epoch']))
   assert chosen['config']==result['selection'][label]['config'] and chosen['selected_epoch']==result['selection'][label]['epoch']
  primary=final_by_label[PRIMARY];gate_selected.append({'cell':cell.name,'gamma':primary['config']['gamma'],'epochs':primary['selected_epoch'],'gate_l2':float(np.linalg.norm(primary['model']['state']['gate'])),'max_delta_from_static':float(np.max(abs(npz[PRIMARY]-npz['static_base'])))})
  feats=np.column_stack((x[:,-20:,0],x[:,-1,1:]))
  with open(cell/'private/hgb.pkl','rb') as ff:ests=pickle.load(ff)
  hh=np.column_stack([e.predict(feats[m['test']]) for e in ests]);gaps['hgb_forward']=max(gaps['hgb_forward'],float(np.max(abs(hh-npz['hgb'][0]))))
  rp=result['selection']['ridge_price'];zz=(x[m['test'],-20:,0]-rp['mean'])/rp['scale'];pp=zz@np.asarray(rp['coefficients']).T+rp['intercept']
  gaps['final_prediction']=max(gaps['final_prediction'],float(np.max(abs(pp-npz['ridge_price'][0]))))
  names=sorted(result['scores']);err=[]
  for name in names:
   ee=stable_errors(npz[name],target);q=(ee.mean(0)/(target**2).mean(0));total=float(q.mean());gaps['reported_Q']=max(gaps['reported_Q'],abs(total-result['scores'][name]['Q']))
   allrows.append({'cell':cell.name,'market':market,'year':year,'model':name,'Q':total,'H1_Q':float(q[0]),'H5_Q':float(q[1]),'H20_Q':float(q[2]),'origins':len(target)});err.append(ee)
  arrays.append((np.stack(err,-1),target**2));counters.append(result['counter'])
 assert max(gaps.values())<1e-7,gaps
 df=pd.DataFrame(allrows);df.to_csv(out/'AUDITED_SCORES.csv',index=False)
 cfg=pd.DataFrame(configs);cfg.to_csv(out/'CONFIGURATIONS.csv',index=False)
 # All history rows, without exposing NN weight arrays or original prices.
 histories=[]
 for cell in cells:
  ts=json.loads(gzip.decompress((cell/'TRACES.json.gz').read_bytes()))
  for i,r in enumerate(ts):histories.append({'cell':cell.name,'record':i,'label':r['label'],'role':r['role'],'history':r['history']})
 # Retain full exact curves locally; public export below can split by cell.
 (out/'FULL_TRAINING_HISTORIES.json.gz').write_bytes(gzip.compress(json.dumps(histories,separators=(',',':')).encode(),mtime=0))
 dump(out/'FINAL_STRUCTURAL_STATES.json',states)
 means=df.groupby('model')['Q'].mean().to_dict();by_market=df.groupby(['market','model'])['Q'].mean().unstack().to_dict('index')
 rng=np.random.default_rng(202610191);intervals=[]
 pidx=names.index(PRIMARY);cidx=[names.index(c) for c in COMP]
 for L in (20,40):
  draws=np.zeros((20000,len(names)));invalid=0
  for err,rw in arrays:
   n=len(err);num=(n+L-1)//L
   for lo in range(0,20000,200):
    hi=min(lo+200,20000);start=rng.integers(0,n,(hi-lo,num));idx=((start[:,:,None]+np.arange(L)[None,None])%n).reshape(hi-lo,-1)[:,:n]
    den=rw[idx].mean(1);invalid+=int(np.any(den<=0,axis=1).sum())
    if invalid:raise ValueError('Nonpositive bootstrap denominator; report without redrawing')
    q=(err[idx].mean(1)/den[:,:,None]).mean(1);draws[lo:hi]+=q/len(arrays)
  effects=(draws[:,cidx]-draws[:,pidx,None])*100
  for j,c in enumerate(COMP):
   delta=(means[c]-means[PRIMARY])*100
   intervals.append({'comparator':c,'block':L,'improvement_RW_percentage_points':delta,'marginal95':np.quantile(effects[:,j],[.025,.975]).tolist(),'family8_individual99_375':np.quantile(effects[:,j],[.003125,.996875]).tolist(),'invalid_draws':invalid})
 total={k:sum(c[k] for c in counters) for k in counters[0]}
 dump(out/'SUMMARY.json',{'mean_Q':means,'market_Q':by_market,'primary':PRIMARY,'counters':total,
                         'training_fits':total['selection_fits']+total['final_fits'],'individual_estimator_fits_including_three_HGB_targets':total['selection_fits']+total['final_fits']+80,
                         'structural_checkpoints_replayed':replay_count,'gru_final_forward_replayed':gru_count,
                         'fits_with_positive_gradients':fit_positive,'structural_readout_solves_in_training':linear_solves,
                         'epoch_zero_final_gradient_models':zero_final,'distinct_market_origins':int(df[df.model=='rw'].origins.sum()),
                         'selected_gate':gate_selected,'intervals':intervals,'all_development':True,
                         'no_new_Tianjin_or_official_CEA_scores':True,'audit_maximum_gaps':gaps,
                         'main_gain_vs_cold_percent':100*(1-means[PRIMARY]/means['cold_duration']),
                         'main_gain_vs_static_percent':100*(1-means[PRIMARY]/means['static_base']),
                         'full_histories_gzip_sha256':digest(out/'FULL_TRAINING_HISTORIES.json.gz')})
 dump(out/'REPLAY_AUDIT.json',{'status':'PASS','method':'NumPy recurrence and augmented SVD for structural checkpoints; forward reload for GRU/HGB; exact source, target and temporal boundary checks','max_gaps':gaps,'structural_checkpoints':replay_count,'gru_final_checkpoints':gru_count,'input_and_trace_identities':identities,'author_identity':'same assistant, separate numerical implementation; not a separate human or blind reviewer'})
 print(json.dumps({'status':'PASS','mean_Q':means,'counters':total,'gaps':gaps},ensure_ascii=False))
if __name__=='__main__':main()
