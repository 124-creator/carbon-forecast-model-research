import sys,json,time,copy
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
from prefix_loader import read_prefix
GROUPS=[slice(0,1),slice(1,5),slice(5,20)]
def project(gs,rng):
    out=[]
    for i in range(3):
        g=gs[i].clone()
        for j in rng.permutation(3):
            dot=torch.dot(g,gs[j]);den=torch.dot(gs[j],gs[j])
            if dot<0 and den>0:g-=dot/den*gs[j]
        out.append(g)
    raw=sum(gs);proj=sum(out);rn=raw.norm();pn=proj.norm()
    if pn>0:proj=proj*(rn/pn)
    elif rn>0:proj=raw.clone()
    return proj,float(rn),float(pn)
def pair_info(gs,limit=None):
    v=[g if limit is None else g[:limit] for g in gs];cs=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        d=v[i].norm()*v[j].norm();cs.append(float(torch.dot(v[i],v[j])/d) if d>0 else 0.)
    return cs

def fit(p,end,seed):
    ids,x,y,tr,va,sd=base.arrays(p,end);torch.manual_seed(seed);rng=np.random.default_rng(seed);m=base.Model('GRUfree');params=list(m.parameters());enc_n=sum(v.numel() for v in m.encoder.parameters());opt=torch.optim.AdamW(params,lr=.003,weight_decay=.001)
    xx=torch.tensor(x/sd,dtype=torch.float32);yy=torch.tensor(y/sd,dtype=torch.float32)
    with torch.no_grad():best=float(((m(xx[va])-yy[va])**2).mean())
    state=copy.deepcopy(m.state_dict());selected=0;wait=0;hist=[]
    for step in range(200):
        opt.zero_grad();error=(m(xx[tr])-yy[tr])**2;loss=error.mean();gs=[]
        for group in GROUPS:
            task=error[:,group].sum()/(len(tr)*20);g=torch.autograd.grad(task,params,retain_graph=True);gs.append(torch.cat([v.reshape(-1) for v in g]))
        full=pair_info(gs);shared=pair_info(gs,enc_n);pg,rn,pn=project(gs,rng);offset=0
        for v in params:
            v.grad=pg[offset:offset+v.numel()].reshape_as(v).clone();offset+=v.numel()
        gn=float(torch.nn.utils.clip_grad_norm_(params,1));opt.step()
        with torch.no_grad():vl=float(((m(xx[va])-yy[va])**2).mean())
        row=dict(loss=float(loss.detach()),validation=vl,shared_cos=shared,full_cos=full,raw_norm=rn,projected_norm_before_match=pn,used_norm=gn)
        if not np.isfinite([row['loss'],vl,rn,pn,gn]+shared+full).all():raise ValueError('nonfinite')
        hist.append(row)
        if vl<best-1e-6:best=vl;state=copy.deepcopy(m.state_dict());selected=step+1;wait=0
        else:wait+=1
        if wait>=20:break
    m.load_state_dict(state)
    return m,sd,dict(epochs=len(hist),selected=selected,history=hist,max_training_target=int(ids[tr[-1]]+20),validation_first=int(ids[va[0]]),max_validation_target=int(ids[va[-1]]+20))

def fixtures():
    for gs in [[torch.tensor([1.,0.]),torch.tensor([0.,1.]),torch.zeros(2)],[torch.tensor([1.,0.]),torch.tensor([-1.,1.]),torch.zeros(2)],[torch.zeros(2)]*3]:
        g,rn,pn=project(gs,np.random.default_rng(17))
        if not torch.isfinite(g).all() or abs(float(g.norm())-rn)>1e-6:raise ValueError('norm fixture')
    gs=[torch.tensor([1.,0.]),torch.tensor([0.,1.]),torch.zeros(2)];g,_,_=project(gs,np.random.default_rng(17))
    if not torch.equal(g,sum(gs)):raise ValueError('no conflict identity')

def main():
    fixtures();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows,_=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31');p=np.array([float(v['author_reported_price']) for v in rows]);dates=np.array([v['date'] for v in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]]);y=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]]);saved=dict(origins=orig,truth=y,years=years);logs=[];causal=[];local=R/'local_only';local.mkdir(exist_ok=True)
    with np.load(R.parent/'local_only/PREDICTIONS.npz',allow_pickle=False) as old:
        if not np.array_equal(y,old['truth']):raise ValueError('matching')
        for seed in [17,29]:saved[f'plain_{seed}']=old[f'GRUfree_{seed}'].copy()
    for seed in [17,29]:
        saved[f'projected_{seed}']=np.full(y.shape,np.nan)
        for block in range(12):
            if sum(v['seconds'] for v in logs)>600:raise RuntimeError('budget incomplete')
            end=459+40*block;ix=np.where((orig-459)//40==block)[0];start=time.perf_counter();m,sd,row=fit(p,end,seed);saved[f'projected_{seed}'][ix]=base.predictions(m,sd,p,orig[ix]);logs.append(dict(seed=seed,block=block,seconds=time.perf_counter()-start,**row))
            if seed==17 and block==0:
                q=p.copy();q[end:]+=10000;n,ss,_=fit(q,end,seed)
                if sd!=ss or not all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items()) or not np.array_equal(base.predictions(m,sd,p,[458]),base.predictions(n,ss,q,[458])):raise ValueError('future refit')
                causal.append(dict(seed=seed,end=end,passed=True))
            np.savez_compressed(local/'PREDICTIONS.npz',**saved);(R/'FIT_LOGS.json').write_text(json.dumps(logs));print(seed,block,row['selected'],round(logs[-1]['seconds'],2),flush=True)
    scores=[]
    for kind in ['plain','projected']:
        cells=[]
        for yr in ['2020','2021']:
            mask=years==yr
            for h in [1,5,20]:
                v=np.mean([np.mean((saved[f'{kind}_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2);cells.append(dict(year=yr,h=h,ratio=float(v)))
        scores.append(dict(kind=kind,score=float(np.mean([v['ratio'] for v in cells])),cells=cells))
    hs=[v for log in logs for v in log['history']];summary=dict(scores=scores,fits=len(logs),origins=len(orig),fit_seconds=sum(v['seconds'] for v in logs),autograd_calls=3*len(hs),steps=len(hs),shared_negative_pairs=sum(sum(c<0 for c in v['shared_cos']) for v in hs),pair_comparisons=3*len(hs),causal=causal,fixtures_passed=True)
    (R/'RESULTS.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
