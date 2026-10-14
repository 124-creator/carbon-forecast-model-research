import sys,json,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
sys.path.insert(0,str(R.parent/'online_group_probe'));import run_online as online
from prefix_loader import read_prefix

def rollout(p,seed,kind,stop=930):
    torch.manual_seed(seed);m,sd=online.initial(p,seed);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.001);pred=[];logs=[]
    for t in range(459,stop):
        if t>=479:
            hs=np.arange(20)
            ids=np.full(20,t-20) if kind=='complete' else t-hs-1
            x=torch.tensor(np.array([np.diff(p[o-60:o+1]) for o in ids])/sd,dtype=torch.float32)
            y=torch.tensor((p[ids+hs+1]-p[ids])/sd,dtype=torch.float32)
            opt.zero_grad();pr=m(x)[torch.arange(20),torch.arange(20)];loss=((pr-y)**2).mean();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));opt.step()
            if not np.isfinite([float(loss.detach()),gn]).all():raise ValueError('nonfinite')
            logs.append(dict(t=t,labels=20,max_target=int(np.max(ids+hs+1)),loss=float(loss.detach()),grad_norm=gn))
        pred.append(base.predictions(m,sd,p,[t])[0])
    return m,np.array(pred),logs

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows,_=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31');p=np.array([float(v['author_reported_price']) for v in rows]);dates=np.array([v['date'] for v in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
    y=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]]);saved=dict(origins=orig,truth=y,years=years);receipts=[];causal=[]
    with np.load(R.parent/'local_only/PREDICTIONS.npz',allow_pickle=False) as z:
        if not np.array_equal(y,z['truth']):raise ValueError('matching')
        for seed in [17,29]:saved[f'slow_{seed}']=z[f'GRUfree_{seed}'].copy()
    for kind in ['complete','partial']:
        for seed in [17,29]:
            start=time.perf_counter();m,pr,logs=rollout(p,seed,kind);saved[f'{kind}_{seed}']=pr[orig-459];saved[f'average_{kind}_{seed}']=(saved[f'slow_{seed}']+pr[orig-459])/2
            receipts.append(dict(kind=kind,seed=seed,seconds=time.perf_counter()-start,updates=len(logs),label_contributions=sum(v['labels'] for v in logs)));(R/f'{kind}_{seed}_LOG.json').write_text(json.dumps(logs))
            if seed==17:
                q=p.copy();q[600:]+=10000;a,x,_=rollout(p,seed,kind,600);b,xx,_=rollout(q,seed,kind,600)
                if not np.array_equal(x,xx) or not all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()):raise ValueError('future replay')
                causal.append(dict(kind=kind,passed=True,through=599,perturbed_from=600))
    scores=[]
    for kind in ['complete','partial','average_complete','average_partial']:
        cells=[]
        for yr in ['2020','2021']:
            mask=years==yr
            for h in [1,5,20]:
                v=np.mean([np.mean((saved[f'{kind}_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2);cells.append(dict(year=yr,h=h,ratio=float(v)))
        scores.append(dict(kind=kind,score=float(np.mean([v['ratio'] for v in cells])),cells=cells))
    local=R/'local_only';local.mkdir(exist_ok=True);np.savez_compressed(local/'PREDICTIONS.npz',**saved);(R/'RESULTS.json').write_text(json.dumps(dict(scores=scores,receipts=receipts,causal=causal,origins=len(orig)),indent=2));print(json.dumps(scores,indent=2))
if __name__=='__main__':main()
