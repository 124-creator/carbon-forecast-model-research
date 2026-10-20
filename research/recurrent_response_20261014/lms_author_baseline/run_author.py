from pathlib import Path
import sys,json,time
from types import SimpleNamespace
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
sys.path.insert(0,str(R/'vendor'));import lms
from prefix_loader import read_prefix
class Wrapper(torch.nn.Module):
    def __init__(self,kind):
        super().__init__();self.model=lms.Model(SimpleNamespace(task_name='long_term_forecast',seq_len=60,label_len=0,pred_len=20,enc_in=1,d_model=16,dropout=0.,channel_independence=1,use_norm=1))
    def forward(self,x):
        levels=x.cumsum(dim=1)[:,:,None]
        return self.model(levels,None,None,None)[:,:,0]-levels[:,-1,0,None]
base.Model=Wrapper

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows,_=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
    p=np.array([float(v['author_reported_price']) for v in rows]);dates=np.array([v['date'] for v in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
    y=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]]);saved=dict(origins=orig,truth=y,years=years);logs=[];causal=[];local=R/'local_only';local.mkdir(exist_ok=True)
    for seed in [17,29]:
        saved[f'lms_{seed}']=np.full(y.shape,np.nan)
        for block in range(12):
            if sum(v['seconds'] for v in logs)>600:raise RuntimeError('budget reached incomplete')
            end=459+40*block;ids=np.where((orig-459)//40==block)[0];start=time.perf_counter();m,sd,row=base.fit(p,end,'author',seed)
            saved[f'lms_{seed}'][ids]=base.predictions(m,sd,p,orig[ids]);logs.append(dict(seed=seed,block=block,seconds=time.perf_counter()-start,**row))
            if seed==17 and block==0:
                q=p.copy();q[end:]+=10000;n,ss,_=base.fit(q,end,'author',seed)
                if sd!=ss or not all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items()) or not np.array_equal(base.predictions(m,sd,p,[458]),base.predictions(n,ss,q,[458])):raise ValueError('future refit')
                causal.append(dict(seed=seed,fit_end=end,passed=True))
            np.savez_compressed(local/'PREDICTIONS.npz',**saved);(R/'FIT_LOGS.json').write_text(json.dumps(logs));print(seed,block,row['selected'],round(logs[-1]['seconds'],2),flush=True)
    cells=[]
    for yr in ['2020','2021']:
        mask=years==yr
        for h in [1,5,20]:
            v=np.mean([np.mean((saved[f'lms_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2);cells.append(dict(year=yr,h=h,ratio=float(v)))
    result=dict(score=float(np.mean([v['ratio'] for v in cells])),cells=cells,fits=len(logs),origins=len(orig),fit_seconds=sum(v['seconds'] for v in logs),causal=causal,complete=len(logs)==24)
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
