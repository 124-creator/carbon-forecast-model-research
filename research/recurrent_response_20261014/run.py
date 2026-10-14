import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,copy
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'torch_segmented_backbone_20261014'))
from prefix_loader import read_prefix
KINDS=['GRUfree','LSTMfree','GRUfixed','GRUglobal','GRUadaptive']

class Model(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.encoder=(nn.LSTM if kind=='LSTMfree' else nn.GRU)(1,16,batch_first=True)
        self.head=nn.Linear(16,20 if kind.endswith('free') else (3 if kind.endswith('adaptive') else 2))
        if kind.endswith('global'):self.speed=nn.Parameter(torch.tensor(-2.6026897))
        self.register_buffer('h',torch.arange(1,21,dtype=torch.float32)[None])
    def forward(self,x):
        z=self.encoder(x[:,:,None])[0][:,-1];v=self.head(z)
        if self.kind.endswith('free'):return v
        tau=6. if self.kind.endswith('fixed') else 2+58*torch.sigmoid(v[:,2:3] if self.kind.endswith('adaptive') else self.speed)
        return v[:,0:1]*self.h+v[:,1:2]*(-torch.expm1(-self.h/tau))

def arrays(p,end):
    ids=np.arange(max(60,end-20-480),end-20)
    x=np.array([np.diff(p[t-60:t+1]) for t in ids]);y=np.array([p[t+1:t+21]-p[t] for t in ids])
    split=len(ids)-60;tr=np.arange(split-20);va=np.arange(split,len(ids))
    sd=max(float(x[tr].std()),1e-6)
    return ids,x,y,tr,va,sd

def fit(p,end,kind,seed):
    ids,x,y,tr,va,sd=arrays(p,end)
    torch.manual_seed(seed);m=Model(kind);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001)
    xx=torch.tensor(x/sd,dtype=torch.float32);yy=torch.tensor(y/sd,dtype=torch.float32)
    with torch.no_grad():best=float(((m(xx[va])-yy[va])**2).mean())
    state=copy.deepcopy(m.state_dict());selected=0;wait=0;hist=[]
    for step in range(200):
        opt.zero_grad();loss=((m(xx[tr])-yy[tr])**2).mean();loss.backward();gn=float(nn.utils.clip_grad_norm_(m.parameters(),1));opt.step()
        with torch.no_grad():vl=float(((m(xx[va])-yy[va])**2).mean())
        hist.append([float(loss.detach()),vl,gn])
        if not np.isfinite(hist[-1]).all():raise ValueError('nonfinite training')
        if vl<best-1e-6:best=vl;state=copy.deepcopy(m.state_dict());selected=step+1;wait=0
        else:wait+=1
        if wait>=20:break
    m.load_state_dict(state)
    return m,sd,dict(epochs=len(hist),selected=selected,history=hist,n=len(ids),max_training_target=int(ids[tr[-1]]+20),validation_first=int(ids[va[0]]),max_validation_target=int(ids[va[-1]]+20),parameters=sum(p.numel() for p in m.parameters()))

def predictions(m,sd,p,orig):
    x=np.array([np.diff(p[t-60:t+1]) for t in orig]);
    with torch.no_grad():return m(torch.tensor(x/sd,dtype=torch.float32)).numpy().astype(float)*sd

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows,audit=read_prefix(R/'../../data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
    p=np.array([float(r['author_reported_price']) for r in rows]);dates=np.array([r['date'] for r in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
    truth=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([d[:4] for d in dates[orig]])
    local=R/'local_only';local.mkdir(exist_ok=True)
    saved=dict(origins=orig,truth=truth,years=years);logs=[];causal=[];elapsed_total=0.
    if (R/'FIT_LOGS.json').exists() and (local/'PREDICTIONS.npz').exists():
        logs=json.loads((R/'FIT_LOGS.json').read_text());elapsed_total=sum(row['seconds'] for row in logs)
        with np.load(local/'PREDICTIONS.npz',allow_pickle=False) as z:
            if not np.array_equal(z['truth'],truth) or not np.array_equal(z['origins'],orig):raise ValueError('resume inputs')
            saved.update({k:z[k].copy() for k in z.files})
        if (R/'CAUSAL.json').exists():causal=json.loads((R/'CAUSAL.json').read_text())
    for kind in KINDS:
        for seed in [17,29]:
            key=f'{kind}_{seed}';saved.setdefault(key,np.full(truth.shape,np.nan))
            for block in range(12):
                ids=np.where((orig-459)//40==block)[0];end=459+40*block
                if any(a['kind']==kind and a['seed']==seed and a['block']==block for a in logs):continue
                if elapsed_total>800:break
                start=time.perf_counter();m,sd,row=fit(p,end,kind,seed);saved[key][ids]=predictions(m,sd,p,orig[ids]);elapsed=time.perf_counter()-start;elapsed_total+=elapsed
                if block==0 and seed==17:
                    q=p.copy();q[end:]+=10000
                    n,ss,_=fit(q,end,kind,seed)
                    equal=all(torch.equal(a,n.state_dict()[k]) for k,a in m.state_dict().items())
                    if not equal or sd!=ss or not np.array_equal(predictions(m,sd,p,[end-1]),predictions(n,ss,q,[end-1])):raise ValueError('future refit')
                    causal.append(dict(kind=kind,seed=seed,fit_end=end,passed=True));(R/'CAUSAL.json').write_text(json.dumps(causal,indent=2))
                np.savez_compressed(local/f'{key}_block{block}_parameters.npz',**{k:v.numpy() for k,v in m.state_dict().items()})
                logs.append(dict(kind=kind,seed=seed,block=block,fit_end=end,seconds=elapsed,**row))
                np.savez_compressed(local/'PREDICTIONS.npz',**saved);(R/'FIT_LOGS.json').write_text(json.dumps(logs))
                print(kind,seed,block,row['epochs'],row['selected'],round(elapsed,2),flush=True)
    scores=[]
    for kind in KINDS:
        if not all(f'{kind}_{s}' in saved and np.isfinite(saved[f'{kind}_{s}']).all() for s in [17,29]):continue
        cells=[]
        for year in ['2020','2021']:
            mask=years==year
            for h in [1,5,20]:
                mse=np.mean([np.mean((saved[f'{kind}_{s}'][mask,h-1]-truth[mask,h-1])**2) for s in [17,29]])
                cells.append(dict(year=year,h=h,ratio=float(mse/np.mean(truth[mask,h-1]**2))))
        scores.append(dict(kind=kind,score=float(np.mean([c['ratio'] for c in cells])),cells=cells))
    result=dict(scores=scores,fits=len(logs),successful_fit_seconds=elapsed_total,complete=len(logs)==120,origins=len(orig),scope='development candidate; novelty unverified')
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
