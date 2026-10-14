from pathlib import Path
import sys,json,time,copy
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
sys.path.insert(0,str(R.parent/'online_group_probe'));import run_online as online
from prefix_loader import read_prefix

def weight(s,f,y,orig):
    out=s.copy()
    for i,t in enumerate(orig):
        for j in range(20):
            ids=np.where(orig+j+1<t)[0][-480:];w=.5
            if len(ids)>=20:
                d=f[ids,j]-s[ids,j];den=float(d@d)
                if den>1e-12:w=float(np.clip(((d@(y[ids,j]-s[ids,j]))/den+.5)/2,0,1))
            out[i,j]=s[i,j]+w*(f[i,j]-s[i,j])
    return out

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows,_=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
    p=np.array([float(r['author_reported_price']) for r in rows]);dates=np.array([r['date'] for r in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
    y=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]])
    saved=dict(origins=orig,truth=y,years=years);logs=[];causal=[];receipts=[];local=R/'local_only';local.mkdir(exist_ok=True)
    with np.load(R.parent/'complementarity_probe/local_only/PREDICTIONS.npz',allow_pickle=False) as z:
        if not np.array_equal(z['truth'],y):raise ValueError('old targets')
        for kind in ['slow','fast','average','delayed_weight']:
            for seed in [17,29]:saved[f'{kind}_{seed}']=z[f'{kind}_{seed}'].copy()
    for seed in [41,53]:
        pred=np.full(y.shape,np.nan);initial_state=None;initial_sd=None
        for block in range(12):
            end=459+40*block;ids=np.where((orig-459)//40==block)[0];start=time.perf_counter()
            m,sd,log=base.fit(p,end,'GRUfree',seed)
            pred[ids]=base.predictions(m,sd,p,orig[ids]);logs.append(dict(seed=seed,block=block,seconds=time.perf_counter()-start,**log))
            if block==0:
                initial_state=copy.deepcopy(m.state_dict());initial_sd=sd
                if seed==41:
                    q=p.copy();q[end:]+=10000;n,ss,_=base.fit(q,end,'GRUfree',seed)
                    if ss!=sd or not all(torch.equal(v,n.state_dict()[k]) for k,v in initial_state.items()):raise ValueError('future fit')
                    causal.append(dict(seed=seed,kind='initial_fit',passed=True))
            (R/'FIT_LOGS.json').write_text(json.dumps(logs));print(seed,block,log['selected'],flush=True)
        def init(pp,unused_seed):
            mm=base.Model('GRUfree');mm.load_state_dict(initial_state);return mm,base.arrays(pp,459)[-1]
        online.initial=init
        start=time.perf_counter();m,pr,steps=online.rollout(p,'ordinary',seed)
        receipts.append(dict(seed=seed,seconds=time.perf_counter()-start,updates=len(steps),labels=sum(v['labels'] for v in steps)))
        (R/f'ONLINE_{seed}_LOG.json').write_text(json.dumps(steps))
        if seed==41:
            q=p.copy();q[600:]+=10000;a,x,_=online.rollout(p,'ordinary',seed,600);b,xx,_=online.rollout(q,'ordinary',seed,600)
            if not np.array_equal(x,xx) or not all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()):raise ValueError('future online')
            causal.append(dict(seed=seed,kind='online_until_599',passed=True))
        saved[f'slow_{seed}']=pred;saved[f'fast_{seed}']=pr[orig-459];saved[f'average_{seed}']=(pred+pr[orig-459])/2;saved[f'delayed_weight_{seed}']=weight(pred,pr[orig-459],y,orig)
        np.savez_compressed(local/'PREDICTIONS.npz',**saved)
    def score(pred):
        return float(np.mean([np.mean((pred[years==yr,h-1]-y[years==yr,h-1])**2)/np.mean(y[years==yr,h-1]**2) for yr in ['2020','2021'] for h in [1,5,20]]))
    scores={k:score(v) for k,v in saved.items() if k not in ['origins','truth','years']}
    groups={kind:dict(mean_seed_loss=float(np.mean([scores[f'{kind}_{s}'] for s in [17,29,41,53]])),prediction_ensemble=score(np.mean([saved[f'{kind}_{s}'] for s in [17,29,41,53]],axis=0))) for kind in ['slow','fast','average','delayed_weight']}
    (R/'RESULTS.json').write_text(json.dumps(dict(scores=scores,groups=groups,fits=len(logs),receipts=receipts,causal=causal,origins=len(orig)),indent=2));print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
