import json,time,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from backbone import Backbone
from prefix_loader import read_prefix
O=Path(__file__).parent
C=json.loads((O/'market_protocol.json').read_text())
def score(pred,y,years):
 return float(np.mean([np.mean((pred[years==yr,j]-y[years==yr,j])**2)/np.mean(y[years==yr,j]**2) for yr in ['2020','2021'] for j in range(3)]))
def main():
 start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 rows,audit=read_prefix(O/'../../data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
 p=np.array([float(r['author_reported_price']) for r in rows]);dates=np.array([r['date'] for r in rows]);H=np.array(C['horizons']);initial=int(np.sum(dates<'2020-01-01'))
 origins=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 if len(origins)!=451:raise ValueError('origins')
 delta=np.diff(p,prepend=p[0]);X=np.zeros((len(p),60))
 for t in range(60,len(p)):X[t]=delta[t-59:t+1]
 truth=p[origins[:,None]+H]-p[origins,None];years=np.array([d[:4] for d in dates[origins]])
 results=[];logs=[];saved=dict(origins=origins,truth=truth,years=years)
 for name in C['models']:
  for seed in (C['seeds'] if name!='linear' else [7]):
   pred=np.full(truth.shape,np.nan)
   for fit in range(initial,len(p),40):
    if time.monotonic()-start>C['budget_seconds']:raise TimeoutError('training budget')
    pos=np.where((origins>=fit)&(origins<fit+40))[0]
    if not len(pos):continue
    ids=np.arange(max(60,fit-20-480),fit-1);valid=ids[ids>=fit-20-60];train=ids[ids<valid[0]-20]
    Y=np.zeros((len(ids),3));mask=ids[:,None]+H<fit
    for j,h in enumerate(H):Y[mask[:,j],j]=p[ids[mask[:,j]]+h]-p[ids[mask[:,j]]]
    tm=np.isin(ids,train);vm=np.isin(ids,valid);sx=max(np.std(X[train]),1e-6);sy=np.array([max(np.std(Y[tm & mask[:,j],j]),1e-6) for j in range(3)])
    xx=torch.tensor(X[ids]/sx,dtype=torch.float64);yy=torch.tensor(Y/sy,dtype=torch.float64);mm=torch.tensor(mask,dtype=torch.float64)
    torch.manual_seed(seed)
    model=Backbone(seed) if name=='SegRNN' else (nn.Sequential(nn.Linear(60,24),nn.Tanh(),nn.Linear(24,3)).double() if name=='MLP' else nn.Linear(60,3).double())
    opt=torch.optim.AdamW(model.parameters(),lr=C['lr'],weight_decay=C['weight_decay']);best=float('inf');state=None;wait=0;hist=[]
    for ep in range(C['max_epochs']):
     opt.zero_grad();err=(model(xx)-yy)**2*mm;loss=err[tm].sum()/mm[tm].sum();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step()
     with torch.no_grad():v=float((((model(xx)-yy)**2*mm)[vm]).sum()/mm[vm].sum())
     hist.append([float(loss.detach()),v,gn])
     if v<best-1e-6:best=v;state=copy.deepcopy(model.state_dict());wait=0;selected=ep+1
     else:wait+=1
     if wait>=C['patience']:break
    if state is None:raise ValueError('nonfinite training')
    model.load_state_dict(state)
    with torch.no_grad():pred[pos]=model(torch.tensor(X[origins[pos]]/sx,dtype=torch.float64)).numpy()*sy
    logs.append(dict(model=name,seed=seed,fit=fit,train_max_target=int(train[-1]+20),validation_max_target=fit-1,epochs=len(hist),selected=selected,history=hist))
   if not np.isfinite(pred).all():raise ValueError('missing prediction')
   key=f'{name}_{seed}';saved[key]=pred;r=dict(model=name,seed=seed,score=score(pred,truth,years));results.append(r);print(r,flush=True)
   np.savez(O/'MARKET_LOCAL_ONLY.npz',**saved);(O/'MARKET_RESULTS.json').write_text(json.dumps(results,indent=2));(O/'MARKET_LOGS.json').write_text(json.dumps(logs))
 (O/'MARKET_RECEIPT.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,audit=audit,origins=len(origins),fits=len(logs),protocol=C),indent=2))
 for name in ['SegRNN','MLP']:print(name,'mean seed loss',np.mean([r['score'] for r in results if r['model']==name]))
if __name__=='__main__':main()
