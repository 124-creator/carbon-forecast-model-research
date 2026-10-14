import sys,json,time,copy
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
from prefix_loader import read_prefix
class Model(base.Model):
    def __init__(self):
        super().__init__('GRUfree');self.aux=torch.nn.Linear(16,1)
    def auxiliary(self,x):
        z=self.encoder(x[:,:,None])[0]
        return ((self.aux(z[:,:-1])[:,:,0]-x[:,1:])**2).mean()

def fit(p,end,seed,weight):
    ids,x,y,tr,va,sd=base.arrays(p,end);torch.manual_seed(seed);m=Model();opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001)
    xx=torch.tensor(x/sd,dtype=torch.float32);yy=torch.tensor(y/sd,dtype=torch.float32)
    with torch.no_grad():best=float(((m(xx[va])-yy[va])**2).mean())
    state=copy.deepcopy(m.state_dict());selected=0;wait=0;hist=[]
    for step in range(200):
        opt.zero_grad();main=((m(xx[tr])-yy[tr])**2).mean();aux=m.auxiliary(xx[tr]) if weight else torch.tensor(0.);loss=main+weight*aux;loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));opt.step()
        with torch.no_grad():vl=float(((m(xx[va])-yy[va])**2).mean())
        hist.append([float(main.detach()),float(aux.detach()),vl,gn])
        if not np.isfinite(hist[-1]).all():raise ValueError('nonfinite')
        if vl<best-1e-6:best=vl;state=copy.deepcopy(m.state_dict());selected=step+1;wait=0
        else:wait+=1
        if wait>=20:break
    m.load_state_dict(state)
    return m,sd,dict(epochs=len(hist),selected=selected,history=hist,main_train_n=len(tr),aux_targets_per_sample=59,max_training_target=int(ids[tr[-1]]+20),validation_first=int(ids[va[0]]),max_validation_target=int(ids[va[-1]]+20),parameters=sum(v.numel() for v in m.parameters()))

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows,_=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31');p=np.array([float(v['author_reported_price']) for v in rows]);dates=np.array([v['date'] for v in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]]);y=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]])
    saved=dict(origins=orig,truth=y,years=years);logs=[];causal=[];local=R/'local_only';local.mkdir(exist_ok=True)
    for kind,w in [('matched',0.),('auxiliary',.1)]:
        for seed in [17,29]:
            saved[f'{kind}_{seed}']=np.full(y.shape,np.nan)
            for block in range(12):
                if sum(v['seconds'] for v in logs)>600:raise RuntimeError('budget incomplete')
                end=459+40*block;ix=np.where((orig-459)//40==block)[0];start=time.perf_counter();m,sd,row=fit(p,end,seed,w);saved[f'{kind}_{seed}'][ix]=base.predictions(m,sd,p,orig[ix]);logs.append(dict(kind=kind,seed=seed,block=block,seconds=time.perf_counter()-start,**row))
                if kind=='auxiliary' and seed==17 and block==0:
                    q=p.copy();q[end:]+=10000;n,ss,_=fit(q,end,seed,w)
                    if sd!=ss or not all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items()) or not np.array_equal(base.predictions(m,sd,p,[458]),base.predictions(n,ss,q,[458])):raise ValueError('future refit')
                    causal.append(dict(kind=kind,seed=seed,end=end,passed=True))
                np.savez_compressed(local/'PREDICTIONS.npz',**saved);(R/'FIT_LOGS.json').write_text(json.dumps(logs));print(kind,seed,block,row['selected'],flush=True)
            if kind=='matched':
                with np.load(R.parent/'local_only/PREDICTIONS.npz',allow_pickle=False) as old:
                    if not np.array_equal(saved[f'{kind}_{seed}'],old[f'GRUfree_{seed}']):raise ValueError('baseline reproduction')
    scores=[]
    for kind in ['matched','auxiliary']:
        cells=[]
        for yr in ['2020','2021']:
            mask=years==yr
            for h in [1,5,20]:
                v=np.mean([np.mean((saved[f'{kind}_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2);cells.append(dict(year=yr,h=h,ratio=float(v)))
        scores.append(dict(kind=kind,score=float(np.mean([v['ratio'] for v in cells])),cells=cells))
    result=dict(scores=scores,origins=len(orig),fits=len(logs),causal=causal,baseline_reproduction=True,fit_seconds=sum(v['seconds'] for v in logs));(R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
