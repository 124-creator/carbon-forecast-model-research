import sys,json,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as backbone
sys.path.insert(0,str(R.parents[1]/'torch_segmented_backbone_20261014'))
from prefix_loader import read_prefix
from torch import nn

KINDS=['ordinary','diagonal','correlated']
def initial(p,seed):
    m=backbone.Model('GRUfree')
    with np.load(R.parent/f'local_only/GRUfree_{seed}_block0_parameters.npz',allow_pickle=False) as z:
        m.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    sd=backbone.arrays(p,459)[-1]
    return m,sd

def precision(pool,hs,kind):
    if kind=='ordinary' or len(pool)<60:return np.eye(len(hs)),0.,1.
    c=np.cov(np.array(pool[-120:]),rowvar=False);diag=np.diag(np.diag(c));c=diag if kind=='diagonal' else .75*diag+.25*c
    c=c[np.ix_(hs,hs)]+1e-4*np.eye(len(hs));w=np.linalg.inv(c);w*=len(hs)/np.trace(w)
    return w,float(np.max(abs(w-np.diag(np.diag(w))))),float(np.linalg.cond(c))

def rollout(p,kind,seed,stop=930):
    torch.manual_seed(seed);m,sd=initial(p,seed);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.001)
    issued={};pool=[];logs=[]
    for t in range(459,stop):
        hs=np.arange(min(20,t-459));orig=t-(hs+1)
        residual=None
        if len(hs):
            inputs=np.array([np.diff(p[o-60:o+1]) for o in orig])/sd
            y=(p[t]-p[orig])/sd
            residual=np.array([y[j]-issued[int(o)][int(h)]/sd for j,(o,h) in enumerate(zip(orig,hs))])
            w,off,condition=precision(pool,hs,kind)
            xt=torch.tensor(inputs,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
            opt.zero_grad();pr=m(xt)[torch.arange(len(hs)),torch.tensor(hs)];error=pr-yt
            loss=error@wt@error/len(hs);loss.backward();gn=float(nn.utils.clip_grad_norm_(m.parameters(),1.))
            before={k:v.detach().clone() for k,v in m.named_parameters()};opt.step()
            delta=float(sum(torch.sum((v-before[k])**2) for k,v in m.named_parameters()).sqrt())
            if not np.isfinite([float(loss.detach()),gn,delta,condition]).all():raise ValueError('nonfinite')
            logs.append(dict(t=t,labels=len(hs),loss=float(loss.detach()),grad_norm=gn,step_norm=delta,offdiag=off,condition=condition,pool_before=len(pool)))
        issued[t]=backbone.predictions(m,sd,p,[t])[0]
        if residual is not None and len(residual)==20:pool.append(residual)
    return m,np.array([issued[t] for t in range(459,stop)]),logs

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows,audit=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
    p=np.array([float(r['author_reported_price']) for r in rows]);dates=np.array([r['date'] for r in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
    truth=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([d[:4] for d in dates[orig]])
    local=R/'local_only';local.mkdir(exist_ok=True);saved=dict(origins=orig,truth=truth,years=years);receipts=[];causal=[]
    with np.load(R.parent/'local_only/PREDICTIONS.npz',allow_pickle=False) as old:
        first=orig[(orig-459)//40==0]
        for seed in [17,29]:
            m,sd=initial(p,seed)
            if not np.array_equal(backbone.predictions(m,sd,p,first),old[f'GRUfree_{seed}'][:len(first)]):raise ValueError('initial reproduction')
    for kind in KINDS:
        for seed in [17,29]:
            start=time.perf_counter();m,pr,logs=rollout(p,kind,seed);seconds=time.perf_counter()-start
            key=f'{kind}_{seed}';saved[key]=pr[orig-459];(R/f'{key}_LOG.json').write_text(json.dumps(logs))
            receipts.append(dict(kind=kind,seed=seed,seconds=seconds,updates=len(logs),label_contributions=sum(a['labels'] for a in logs),clip_count=sum(a['grad_norm']>1 for a in logs),max_condition=max(a['condition'] for a in logs),non_diagonal_updates=sum(a['offdiag']>0 for a in logs)))
            if seed==17:
                q=p.copy();q[600:]+=10000
                a,x,_=rollout(p,kind,seed,stop=600);b,y,_=rollout(q,kind,seed,stop=600)
                equal=all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
                if not equal or not np.array_equal(x,y):raise ValueError('future online replay')
                causal.append(dict(kind=kind,seed=seed,unchanged_through=599,perturbed_from=600,passed=True))
            np.savez_compressed(local/'PREDICTIONS.npz',**saved);print(key,round(seconds,2),receipts[-1],flush=True)
    scores=[]
    for kind in KINDS:
        cells=[]
        for year in ['2020','2021']:
            mask=years==year
            for h in [1,5,20]:
                mse=np.mean([np.mean((saved[f'{kind}_{s}'][mask,h-1]-truth[mask,h-1])**2) for s in [17,29]])
                cells.append(dict(year=year,h=h,ratio=float(mse/np.mean(truth[mask,h-1]**2))))
        scores.append(dict(kind=kind,score=float(np.mean([c['ratio'] for c in cells])),cells=cells))
    result=dict(scores=scores,receipts=receipts,causal=causal,origins=len(orig),scope='exploratory online weighted loss; novelty unverified; not DSOF reproduction')
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(scores,indent=2))
if __name__=='__main__':main()
