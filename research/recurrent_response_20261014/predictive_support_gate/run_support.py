import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
from prefix_loader import read_prefix
MODES=['raw','single','shared','freeze']

def support(h,y):
    h=h-h.mean(0);y=y-y.mean(0)
    b=np.linalg.solve(h.T@h+0.1*np.eye(h.shape[1]),h.T@y)
    u,s,_=np.linalg.svd(b,full_matrices=False)
    return u[:, :min(3,int(np.sum(s>1e-10))) ]

def intersection(a,b):
    if not a.shape[1] or not b.shape[1]:return np.zeros((a.shape[0],0)),[]
    _,s,v=np.linalg.svd(a.T@b,full_matrices=False)
    return b@v.T[:,s>=0.9],s.tolist()

def jacobian(m,x):
    z=m.encoder(x[:,:,None])[0][:,-1].mean(0);ps=list(m.encoder.parameters());rows=[]
    for j in range(16):
        g=torch.autograd.grad(z[j],ps,retain_graph=True)
        rows.append(torch.cat([v.reshape(-1) for v in g]))
    return torch.stack(rows)

def lift(j,q):
    if not q.shape[1]:return torch.zeros((j.shape[1],0),dtype=j.dtype)
    a=j.detach().numpy().astype(float).T@q
    u,s,_=np.linalg.svd(a,full_matrices=False)
    return torch.tensor(u[:,s>1e-10],dtype=j.dtype)

def project(g,b):
    if not b.shape[1]:return torch.zeros_like(g)
    v=b@(b.T@g);n=v.norm()
    return v*(g.norm()/n) if n>1e-12 else torch.zeros_like(g)

def fixtures():
    e=np.eye(16);q,s=intersection(e[:,:3],e[:,[0,3,4]])
    if q.shape[1]!=1 or not np.allclose(q@q.T,np.outer(e[:,0],e[:,0]),atol=1e-12):raise ValueError('exact intersection')
    z,_=intersection(e[:,:3],e[:,3:6])
    if z.shape[1]:raise ValueError('orthogonal spaces')
    z,_=intersection(np.zeros((16,0)),e[:,:3])
    if z.shape[1]:raise ValueError('zero rank')
    a=np.diag([3.,2.,1.]+[0.]*13);u=np.linalg.svd(a,full_matrices=False)[0][:,:3];v=np.linalg.svd(-a,full_matrices=False)[0][:,:3];q,_=intersection(u,v)
    if not np.allclose(q@q.T,u@u.T,atol=1e-12):raise ValueError('sign counterexample')
    b=torch.tensor(e[:,:3],dtype=torch.float64);g=torch.arange(16,dtype=torch.float64)
    pg=project(g,b)
    if abs(float(pg.norm()-g.norm()))>1e-10 or float((pg-b@(b.T@pg)).norm())>1e-10:raise ValueError('projection fixture')
    torch.manual_seed(17);m=base.Model('GRUfree').double();x=torch.randn(4,6,dtype=torch.float64);j=jacobian(m,x);ps=list(m.encoder.parameters());d=torch.randn(j.shape[1],dtype=torch.float64);d/=d.norm();states=copy.deepcopy(m.state_dict());values=[]
    for sign in [-1,1]:
        m.load_state_dict(states);offset=0
        with torch.no_grad():
            for p in ps:p.add_(sign*1e-5*d[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
            values.append(m.encoder(x[:,:,None])[0][:,-1].mean(0))
    err=float(((values[1]-values[0])/2e-5-j@d).abs().max())
    if err>1e-7:raise ValueError('Jacobian finite difference')
    return dict(passed=True,jacobian_max_error=err,sign_reversal_same_subspace=True,sign_reversal_interpretation='common support does not prove stable conditional response')

def adapt(p,end,initial,mode):
    ids,x,y,tr,va,sd=base.arrays(p,end)
    m=base.Model('GRUfree');m.load_state_dict(initial)
    xx=torch.tensor(x/sd,dtype=torch.float32);yy=torch.tensor(y/sd,dtype=torch.float32)
    mid=len(tr)//2;a=tr[:mid-20];b=tr[mid:]
    if ids[a[-1]]+20>=ids[b[0]]:raise ValueError('internal purge')
    with torch.no_grad():h=m.encoder(xx[tr,:,None])[0][:,-1].numpy().astype(float)
    qa=support(h[:len(a)],y[a]/sd);qb=support(h[mid:],y[b]/sd);qs,cos=intersection(qa,qb)
    ps=list(m.encoder.parameters());size=sum(v.numel() for v in ps)
    if mode in ['single','shared']:
        ix=tr[np.linspace(0,len(tr)-1,16,dtype=int)];j=jacobian(m,xx[ix]);basis=lift(j,qb if mode=='single' else qs)
    else:basis=None
    opt=torch.optim.SGD(m.parameters(),lr=.003)
    with torch.no_grad():best=float(((m(xx[va])-yy[va])**2).mean())
    selected=0;state=copy.deepcopy(m.state_dict());history=[]
    for step in range(20):
        opt.zero_grad();loss=((m(xx[tr])-yy[tr])**2).mean();loss.backward()
        g=torch.cat([v.grad.reshape(-1) for v in ps]);raw=float(g.norm())
        if mode=='freeze':g=torch.zeros_like(g)
        elif basis is not None:g=project(g,basis)
        offset=0
        for v in ps:v.grad=g[offset:offset+v.numel()].reshape_as(v).clone();offset+=v.numel()
        used=float(g.norm());norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1));opt.step()
        with torch.no_grad():vl=float(((m(xx[va])-yy[va])**2).mean())
        if not np.isfinite([float(loss.detach()),vl,raw,used,norm]).all():raise ValueError('nonfinite')
        history.append(dict(loss=float(loss.detach()),validation=vl,raw_encoder_norm=raw,used_encoder_norm=used))
        if vl<best-1e-6:best=vl;selected=step+1;state=copy.deepcopy(m.state_dict())
    m.load_state_dict(state)
    return m,sd,dict(mode=mode,selected=selected,epochs=20,encoder_parameters=size,single_rank=qb.shape[1],shared_rank=qs.shape[1],parameter_rank=None if basis is None else basis.shape[1],principal_cos=cos,max_training_target=int(ids[tr[-1]]+20),validation_first=int(ids[va[0]]),max_validation_target=int(ids[va[-1]]+20),internal_early_max_target=int(ids[a[-1]]+20),internal_late_first=int(ids[b[0]]),history=history)

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);fx=fixtures()
    rows,audit=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31');p=np.array([float(v['author_reported_price']) for v in rows]);dates=np.array([v['date'] for v in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]]);y=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]])
    saved=dict(origins=orig,truth=y,years=years);logs=[];causal=[];local=R/'local_only';local.mkdir(exist_ok=True);elapsed=0.
    with np.load(R.parent/'local_only/PREDICTIONS.npz',allow_pickle=False) as old:
        if not np.array_equal(y,old['truth']) or not np.array_equal(orig,old['origins']):raise ValueError('audited baseline targets')
        for seed in [17,29]:saved[f'initial_{seed}']=old[f'GRUfree_{seed}'].copy()
    for seed in [17,29]:
        for mode in MODES:saved[f'{mode}_{seed}']=np.full(y.shape,np.nan)
        for block in range(12):
            end=459+40*block;ix=np.where((orig-459)//40==block)[0]
            with np.load(R.parent/f'local_only/GRUfree_{seed}_block{block}_parameters.npz',allow_pickle=False) as z:initial={k:torch.from_numpy(z[k].copy()) for k in z.files}
            check=base.Model('GRUfree');check.load_state_dict(initial);sd=base.arrays(p,end)[-1]
            if not np.array_equal(base.predictions(check,sd,p,orig[ix]),saved[f'initial_{seed}'][ix]):raise ValueError('initialization reproduction')
            for mode in MODES:
                if elapsed>600:raise RuntimeError('budget incomplete')
                start=time.perf_counter();m,sd,row=adapt(p,end,initial,mode);seconds=time.perf_counter()-start;elapsed+=seconds;saved[f'{mode}_{seed}'][ix]=base.predictions(m,sd,p,orig[ix]);logs.append(dict(seed=seed,block=block,end=end,seconds=seconds,**row))
                if seed==17 and block==0:
                    q=p.copy();q[end:]+=10000;n,ss,nrow=adapt(q,end,initial,mode)
                    if sd!=ss or row!=nrow or not all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items()) or not np.array_equal(base.predictions(m,sd,p,[458]),base.predictions(n,ss,q,[458])):raise ValueError('future adaptation replay')
                    causal.append(dict(mode=mode,end=end,passed=True,scope='additional adaptation only; fixed audited initialization'))
                np.savez_compressed(local/'PREDICTIONS.npz',**saved);(R/'FIT_LOGS.json').write_text(json.dumps(logs));print(seed,block,mode,row['shared_rank'],row['selected'],flush=True)
    scores={}
    for mode in ['initial']+MODES:
        cells=[]
        for yr in ['2020','2021']:
            mask=years==yr
            for h in [1,5,20]:
                v=np.mean([np.mean((saved[f'{mode}_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2);cells.append(dict(year=yr,h=h,ratio=float(v)))
        scores[mode]=dict(score=float(np.mean([v['ratio'] for v in cells])),cells=cells)
    comparisons=['initial','raw','single','freeze'];gate=all(scores['shared']['score']<scores[c]['score'] for c in comparisons) and all(np.mean([v['ratio'] for v in scores['shared']['cells'] if v['year']==yr])<np.mean([v['ratio'] for v in scores[c]['cells'] if v['year']==yr]) for c in comparisons for yr in ['2020','2021'])
    result=dict(scores=scores,fits=len(logs),origins=len(orig),fit_seconds=elapsed,fixtures=fx,causal=causal,audit=audit,development_gate=gate,novelty='unqualified',initialization='reuse audited current-session numeric GRU checkpoints; not fresh initial training',extra_jacobian_calls=16*48)
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
