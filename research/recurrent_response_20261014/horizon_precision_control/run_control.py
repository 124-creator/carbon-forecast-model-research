import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
from prefix_loader import read_prefix

def precision(y):
 q=np.mean(y*y,axis=0);v=1/np.maximum(q,1e-8);return v/np.mean(v)

def fit(p,end,seed):
 ids,x,y,tr,va,sd=base.arrays(p,end);w=precision(y[tr]/sd)
 torch.manual_seed(seed);m=base.Model('GRUfree');opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001)
 xx=torch.tensor(x/sd,dtype=torch.float32);yy=torch.tensor(y/sd,dtype=torch.float32);ww=torch.tensor(w,dtype=torch.float32)
 with torch.no_grad():best=float(((m(xx[va])-yy[va])**2).mean())
 state=copy.deepcopy(m.state_dict());selected=0;wait=0;history=[]
 for step in range(200):
  opt.zero_grad();error=(m(xx[tr])-yy[tr])**2;loss=(error*ww).mean();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));opt.step()
  with torch.no_grad():ve=(m(xx[va])-yy[va])**2;vl=float(ve.mean());wv=float((ve*ww).mean())
  history.append(dict(weighted_train=float(loss.detach()),raw_train=float(error.detach().mean()),raw_validation=vl,weighted_validation_diagnostic_only=wv,gradient_norm=norm))
  if not np.isfinite(list(history[-1].values())).all():raise ValueError('nonfinite')
  if vl<best-1e-6:best=vl;state=copy.deepcopy(m.state_dict());selected=step+1;wait=0
  else:wait+=1
  if wait>=20:break
 m.load_state_dict(state)
 return m,sd,dict(epochs=len(history),selected=selected,history=history,weights=w.tolist(),weight_max_target=int(ids[tr[-1]]+20),validation_first=int(ids[va[0]]),max_validation_target=int(ids[va[-1]]+20),n=len(ids),parameters=sum(v.numel() for v in m.parameters()))

def fixtures():
 e=torch.arange(40,dtype=torch.float32).reshape(2,20)
 if not torch.equal((e*torch.ones(20)).mean(),e.mean()):raise ValueError('uniform loss identity')
 w=precision(np.ones((3,20)))
 if not np.array_equal(w,np.ones(20)):raise ValueError('uniform precision')
 w=precision(np.zeros((3,20)))
 if not np.isfinite(w).all() or abs(w.mean()-1)>1e-12:raise ValueError('zero moment finite')
 return dict(passed=True,uniform_loss_identity=True,zero_moment_finite=True)

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);fx=fixtures()
 rows,audit=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31');p=np.array([float(v['author_reported_price']) for v in rows]);dates=np.array([v['date'] for v in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]]);y=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]])
 saved=dict(origins=orig,truth=y,years=years);logs=[];causal=[];reference=[];elapsed=0.;local=R/'local_only';local.mkdir(exist_ok=True)
 with np.load(R.parent/'local_only/PREDICTIONS.npz',allow_pickle=False) as z:
  if not np.array_equal(y,z['truth']) or not np.array_equal(orig,z['origins']):raise ValueError('baseline inputs')
  for seed in [17,29]:saved[f'plain_{seed}']=z[f'GRUfree_{seed}'].copy()
 for seed in [17,29]:
  start=time.perf_counter();m,sd,_=base.fit(p,459,'GRUfree',seed);ix=np.where((orig-459)//40==0)[0]
  if not np.array_equal(base.predictions(m,sd,p,orig[ix]),saved[f'plain_{seed}'][ix]):raise ValueError('plain representative refit')
  reference.append(dict(seed=seed,end=459,seconds=time.perf_counter()-start,passed=True,scope='one initial block per seed, not all blocks'))
  saved[f'weighted_{seed}']=np.full(y.shape,np.nan)
  for block in range(12):
   if elapsed>600:raise RuntimeError('budget incomplete')
   end=459+40*block;ix=np.where((orig-459)//40==block)[0];start=time.perf_counter();m,sd,row=fit(p,end,seed);seconds=time.perf_counter()-start;elapsed+=seconds;saved[f'weighted_{seed}'][ix]=base.predictions(m,sd,p,orig[ix]);logs.append(dict(seed=seed,block=block,end=end,seconds=seconds,**row))
   if seed==17 and block==0:
    q=p.copy();q[end:]+=10000;start=time.perf_counter();n,ss,other=fit(q,end,seed);extra=time.perf_counter()-start
    if sd!=ss or row!=other or not all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items()) or not np.array_equal(base.predictions(m,sd,p,[458]),base.predictions(n,ss,q,[458])):raise ValueError('future refit including weights')
    causal.append(dict(seed=seed,end=end,seconds=extra,passed=True,scope='representative fresh training, scale and horizon weight refit'))
   np.savez_compressed(local/'PREDICTIONS.npz',**saved);(R/'FIT_LOGS.json').write_text(json.dumps(logs));print(seed,block,row['selected'],round(seconds,2),flush=True)
 scores={};seeds={}
 for mode in ['plain','weighted']:
  cells=[];seeds[mode]={}
  for year in ['2020','2021']:
   mask=years==year
   for h in [1,5,20]:
    ratio=np.mean([np.mean((saved[f'{mode}_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2);cells.append(dict(year=year,h=h,ratio=float(ratio)))
  scores[mode]=dict(score=float(np.mean([v['ratio'] for v in cells])),cells=cells)
  for seed in [17,29]:seeds[mode][str(seed)]=float(np.mean([np.mean((saved[f'{mode}_{seed}'][years==yr,h-1]-y[years==yr,h-1])**2)/np.mean(y[years==yr,h-1]**2) for yr in ['2020','2021'] for h in [1,5,20]]))
 gate=scores['weighted']['score']<scores['plain']['score'] and all(np.mean([v['ratio'] for v in scores['weighted']['cells'] if v['year']==yr])<np.mean([v['ratio'] for v in scores['plain']['cells'] if v['year']==yr]) for yr in ['2020','2021'])
 result=dict(scores=scores,seed_scores=seeds,fits=len(logs),origins=len(orig),fit_seconds=elapsed,reference_refits=reference,causal=causal,fixtures=fx,audit=audit,development_gate=gate,novelty='ordinary loss weighting control, not model innovation',validation_rule='unchanged raw all20 MSE; no horizon-weight selection on validation')
 (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
