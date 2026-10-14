import sys,json,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
from prefix_loader import read_prefix

def load(p,seed,block):
    m=base.Model('GRUfree')
    with np.load(R.parent/f'local_only/GRUfree_{seed}_block{block}_parameters.npz',allow_pickle=False) as z:
        m.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    return m,base.arrays(p,459+40*block)[-1]

def features(m,sd,p,ids):
    x=torch.tensor(np.array([np.diff(p[o-60:o+1]) for o in ids])/sd,dtype=torch.float32)
    with torch.no_grad():
        h=m.encoder(x[:,:,None])[0][:,-1]
        y=m.head(h).numpy().astype(float)*sd
    return np.column_stack([h.numpy().astype(float),np.ones(len(ids))]),y

def rollout(p,seed,kind,stop=930):
    m,sd=load(p,seed,0);a=np.zeros((17,20));pred=[];logs=[];updates=0
    for t in range(459,stop):
        if t>459 and (t-459)%40==0:
            old=m;oss=sd;m,sd=load(p,seed,(t-459)//40)
            ids=np.arange(max(60,t-120),t);xo,_=features(old,oss,p,ids);xn,_=features(m,sd,p,ids)
            target=xo@a
            if kind=='reset':a=np.zeros_like(a)
            elif kind=='ridge':a=np.linalg.solve(xn.T@xn/len(ids)+.01*np.eye(17),xn.T@target/len(ids))
            mismatch=float(np.mean((xn@a-target)**2))
            logs.append(dict(t=t,anchor_max=int(ids[-1]),mismatch=mismatch,kind=kind))
        hs=np.arange(min(20,t-459));ids=t-hs-1
        if len(ids):
            x,y=features(m,sd,p,ids);target=p[t]-p[ids];err=(y+x@a)[np.arange(len(ids)),hs]-target
            g=np.zeros_like(a)
            for j,h in enumerate(hs):g[:,h]+=2*x[j]*err[j]/len(ids)
            norm=np.linalg.norm(g);a-=.001*g/max(1,norm);updates+=1
        x,y=features(m,sd,p,[t]);pred.append((y+x@a)[0])
    return np.array(pred),a,logs,updates

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows,_=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
    p=np.array([float(r['author_reported_price']) for r in rows]);dates=np.array([r['date'] for r in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
    truth=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]])
    saved=dict(origins=orig,truth=truth,years=years);receipts=[];causal=[];scores=[]
    for kind in ['reset','copy','ridge']:
        for seed in [17,29]:
            start=time.perf_counter();pr,a,logs,n=rollout(p,seed,kind);saved[f'{kind}_{seed}']=pr[orig-459]
            if not np.isfinite(pr).all():raise ValueError('nonfinite')
            receipts.append(dict(kind=kind,seed=seed,seconds=time.perf_counter()-start,updates=n,refreshes=logs))
            if seed==17:
                q=p.copy();q[600:]+=10000
                aa,wa,_,_=rollout(p,seed,kind,600);bb,wb,_,_=rollout(q,seed,kind,600)
                if not np.array_equal(aa,bb) or not np.array_equal(wa,wb):raise ValueError('future replay')
                causal.append(dict(kind=kind,through=599,perturbed_from=600,passed=True))
        cells=[]
        for yr in ['2020','2021']:
            mask=years==yr
            for h in [1,5,20]:
                v=np.mean([np.mean((saved[f'{kind}_{s}'][mask,h-1]-truth[mask,h-1])**2) for s in [17,29]])/np.mean(truth[mask,h-1]**2)
                cells.append(dict(year=yr,h=h,ratio=float(v)))
        scores.append(dict(kind=kind,score=float(np.mean([v['ratio'] for v in cells])),cells=cells))
    local=R/'local_only';local.mkdir(exist_ok=True);np.savez_compressed(local/'PREDICTIONS.npz',**saved)
    (R/'RESULTS.json').write_text(json.dumps(dict(scores=scores,receipts=receipts,causal=causal,origins=len(orig)),indent=2));print(json.dumps(scores,indent=2))
if __name__=='__main__':main()
