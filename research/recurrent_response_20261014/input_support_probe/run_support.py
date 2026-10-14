import sys,json,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
from prefix_loader import read_prefix
original_arrays=base.arrays
LENGTH=60

def arrays(p,end):
    ids,x,y,tr,va,sd=original_arrays(p,end);x=x[:,-LENGTH:];sd=max(float(x[tr].std()),1e-6)
    return ids,x,y,tr,va,sd
base.arrays=arrays

def predict(m,sd,p,orig):
    x=np.array([np.diff(p[t-60:t+1])[-LENGTH:] for t in orig])
    with torch.no_grad():return m(torch.tensor(x/sd,dtype=torch.float32)).numpy().astype(float)*sd

def main():
    global LENGTH
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows,_=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31');p=np.array([float(v['author_reported_price']) for v in rows]);dates=np.array([v['date'] for v in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]]);y=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]])
    saved=dict(origins=orig,truth=y,years=years);logs=[];causal=[];local=R/'local_only';local.mkdir(exist_ok=True)
    with np.load(R.parent/'local_only/PREDICTIONS.npz',allow_pickle=False) as z:
        if not np.array_equal(y,z['truth']):raise ValueError('matching')
        for seed in [17,29]:saved[f'L60_{seed}']=z[f'GRUfree_{seed}'].copy()
    for length in [10,20]:
        LENGTH=length
        for seed in [17,29]:
            key=f'L{length}_{seed}';saved[key]=np.full(y.shape,np.nan)
            for block in range(12):
                if sum(v['seconds'] for v in logs)>600:raise RuntimeError('incomplete budget')
                end=459+40*block;ix=np.where((orig-459)//40==block)[0];start=time.perf_counter();m,sd,row=base.fit(p,end,'GRUfree',seed);saved[key][ix]=predict(m,sd,p,orig[ix]);logs.append(dict(length=length,seed=seed,block=block,seconds=time.perf_counter()-start,**row))
                if seed==17 and block==0:
                    q=p.copy();q[end:]+=10000;n,ss,_=base.fit(q,end,'GRUfree',seed)
                    if sd!=ss or not all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items()) or not np.array_equal(predict(m,sd,p,[458]),predict(n,ss,q,[458])):raise ValueError('future refit')
                    causal.append(dict(length=length,seed=seed,end=end,passed=True))
                np.savez_compressed(local/'PREDICTIONS.npz',**saved);(R/'FIT_LOGS.json').write_text(json.dumps(logs));print(length,seed,block,row['selected'],flush=True)
    scores=[]
    for length in [10,20,60]:
        cells=[]
        for yr in ['2020','2021']:
            mask=years==yr
            for h in [1,5,20]:
                v=np.mean([np.mean((saved[f'L{length}_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2);cells.append(dict(year=yr,h=h,ratio=float(v)))
        scores.append(dict(length=length,score=float(np.mean([v['ratio'] for v in cells])),cells=cells))
    r=dict(scores=scores,fits=len(logs),origins=len(orig),causal=causal,fit_seconds=sum(v['seconds'] for v in logs));(R/'RESULTS.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
