"""One predeclared causal decomposition architecture and its learning rules.

No downloads, checkpoint loading, or whole-file price reads. Temporal optimization
uses an exact derivative through ONE clipped SGD step, not through prior outer
iterations. No claim of a converged bilevel solution or novel meta-learning.
"""
from __future__ import annotations
import copy, csv, hashlib, io, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.func import functional_call

SEEDS=(17,29,41,53)
MODES=('S_ERM','D_ERM','S_TEMP','D_TEMP','D_JOINT','GRU')
PREFIX_SHA='1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'


def read_prefix(path: str|Path, check_hash: bool=True):
    """Request exactly 951 newline-terminated lines, never the next byte."""
    lines=[]
    with Path(path).open('rb',buffering=0) as f:
        for _ in range(951):
            line=bytearray()
            while True:
                b=f.read(1)
                if not b: raise ValueError('incomplete permitted prefix')
                line+=b
                if b==b'\n': break
            lines.append(bytes(line))
    raw=b''.join(lines)
    digest=hashlib.sha256(raw).hexdigest()
    if check_hash and digest!=PREFIX_SHA: raise ValueError('prefix hash mismatch')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    dates=np.array([r['date'] for r in rows]); p=np.array([float(r['author_reported_price']) for r in rows])
    if len(rows)!=950 or dates[-1]!='2021-12-31' or not np.all(dates[1:]>dates[:-1]):
        raise ValueError('date/prefix contract')
    if not np.isfinite(p).all() or np.any(p<=0): raise ValueError('invalid prices')
    return p,dates,dict(records=950,prefix_bytes=len(raw),prefix_sha256=digest,last_date=str(dates[-1]),next_byte_requested=False,metadata_features_used=False)


def backward_mean(x: torch.Tensor, k: int):
    if x.ndim!=2 or k<1: raise ValueError('expected batch x length')
    L=x.shape[1]
    ends=torch.arange(1,L+1,device=x.device)
    starts=(ends-k).clamp(min=0)
    cs=torch.cat([torch.zeros_like(x[:,:1]),x.cumsum(1)],1)
    return (cs[:,ends]-cs[:,starts])/(ends-starts).to(x.dtype)


def statistics(x):
    means=torch.stack([backward_mean(x,k) for k in (1,5,20)],-1)
    def lag(v): return torch.cat([torch.zeros_like(v[:,:1]),v[:,:-1]],1)
    context=torch.stack([lag(x),lag(backward_mean(x,5)),lag(backward_mean(x.square(),20))],-1)
    return means,context


class Decomposition(nn.Module):
    def __init__(self,dynamic:bool):
        super().__init__();self.dynamic=dynamic
        self.phi=nn.Parameter(torch.zeros(8 if dynamic else 2))
    def forward(self,x):
        means,q=statistics(x)
        if self.dynamic:
            logits=q@self.phi[:6].reshape(2,3).T+self.phi[6:]
        else:
            logits=self.phi[None,None,:].expand(x.shape[0],x.shape[1],2)
        weights=torch.cat([logits,torch.zeros_like(logits[:,:,:1])],-1).softmax(-1)
        c=(means*weights).sum(-1)
        return c,x-c,weights


class DualGRU(nn.Module):
    def __init__(self,dynamic:bool):
        super().__init__()
        self.router=Decomposition(dynamic)
        self.c_gru=nn.GRU(1,10,batch_first=True)
        self.d_gru=nn.GRU(1,10,batch_first=True)
        self.head=nn.Linear(20,20)
    def forward(self,x):
        c,d,_=self.router(x)
        hc=self.c_gru(c.unsqueeze(-1))[0][:,-1]
        hd=self.d_gru(d.unsqueeze(-1))[0][:,-1]
        return self.head(torch.cat([hc,hd],-1))


class OrdinaryGRU(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(1,16,batch_first=True);self.head=nn.Linear(16,20)
    def forward(self,x): return self.head(self.gru(x.unsqueeze(-1))[0][:,-1])


def make_model(mode,seed,dtype=torch.float32):
    torch.manual_seed(seed)
    m=OrdinaryGRU() if mode=='GRU' else DualGRU(mode.startswith('D'))
    return m.to(dtype=dtype)


def training_arrays(p,end):
    if end>len(p) or end<200: raise ValueError('bad cutoff')
    # A private prefix makes accidental use of later prices structurally harder.
    p=np.asarray(p[:end],dtype=float)
    ids=np.arange(max(60,end-20-480),end-20)
    x=np.array([np.diff(p[t-60:t+1]) for t in ids])
    y=np.array([p[t+1:t+21]-p[t] for t in ids])
    split=len(ids)-60
    F=np.arange(split-20); C=np.arange(split,len(ids))
    a=int(np.floor(.65*(len(F)-20)))
    A=F[:a]; B=F[a+20:]; pool=np.concatenate([A,B])
    if min(len(A),len(B),len(C))<20: raise ValueError('insufficient nested support')
    if ids[A[-1]]+20>=ids[B[0]] or ids[F[-1]]+20>=ids[C[0]]: raise ValueError('purge violation')
    sd=max(float(x[A].std()),1e-6)
    receipt=dict(end_exclusive=int(end),n_origins=len(ids),n_F=len(F),n_A=len(A),n_B=len(B),n_C=len(C),
        A_last_origin=int(ids[A[-1]]),A_last_target=int(ids[A[-1]]+20),B_first_origin=int(ids[B[0]]),
        B_last_target=int(ids[B[-1]]+20),F_last_target=int(ids[F[-1]]+20),C_first_origin=int(ids[C[0]]),
        C_last_target=int(ids[C[-1]]+20),scale=sd,scale_last_price_index=int(ids[A[-1]]),scale_source='A inputs only')
    return ids,x,y,F,C,A,B,pool,sd,receipt


def theta_items(m): return [(n,p) for n,p in m.named_parameters() if n!='router.phi']


def differentiable_clip(grads,cap=1.):
    norm=torch.sqrt(sum(g.square().sum() for g in grads)+1e-24)
    factor=(cap/(norm+1e-12)).clamp(max=1.)
    return [g*factor for g in grads],norm


def temporal_objective(m,xA,yA,xB,yB,eta=.01,penalty=.001):
    """Exact one-step objective, including the derivative of gradient clipping."""
    items=theta_items(m)
    la=(m(xA)-yA).square().mean()
    grads=torch.autograd.grad(la,[p for _,p in items],create_graph=True)
    grads,norm=differentiable_clip(grads)
    upd={n:p-eta*g for (n,p),g in zip(items,grads)}
    params={**upd,'router.phi':m.router.phi}
    lb=(functional_call(m,params,(xB,),strict=True)-yB).square().mean()
    return lb+penalty*m.router.phi.square().sum(),upd,la,lb,norm


def numeric_state(m): return {n:p.detach().cpu().numpy().copy() for n,p in m.state_dict().items()}


def state_digest(m):
    d=hashlib.sha256()
    for n,p in sorted(m.state_dict().items()):d.update(n.encode());d.update(p.detach().numpy().tobytes())
    return d.hexdigest()


def _check_finite(values):
    if not np.isfinite(values).all(): raise FloatingPointError('nonfinite training')


def fit_model(p,end,mode,seed,limits=None):
    """Fits only permitted training prefix; validation C is already mature."""
    wall=time.perf_counter();cpu=time.process_time()
    ids,x,y,F,C,A,B,pool,sd,receipt=training_arrays(p,end)
    dtype=torch.float32
    xx=torch.as_tensor(x/sd,dtype=dtype);yy=torch.as_tensor(y/sd,dtype=dtype)
    lim=dict(warmup=20,outer=40,erm=120,final=200,joint=340,patience=20)
    if limits:lim.update(limits) # tests only; official runner does not override
    m=make_model(mode,seed)
    original={n:v.detach().clone() for n,v in m.named_parameters() if n!='router.phi'}
    initial_digest=state_digest(m)
    warm_hist=[];route_hist=[];final_hist=[];visits=0;phase_seconds={}
    is_factorial=mode in ('S_ERM','D_ERM','S_TEMP','D_TEMP')
    theta=[p for _,p in theta_items(m)]
    if is_factorial:
        mark=time.perf_counter()
        opt=torch.optim.AdamW(theta,lr=.003,weight_decay=.001)
        for step in range(lim['warmup']):
            m.zero_grad(set_to_none=True)
            loss=(m(xx[A])-yy[A]).square().mean();loss.backward()
            gn=float(nn.utils.clip_grad_norm_(theta,1));opt.step();visits+=len(A)
            row=[float(loss.detach()),gn];_check_finite(row);warm_hist.append(row)
        phase_seconds['warmup']=time.perf_counter()-mark
        mark=time.perf_counter()
        optphi=torch.optim.Adam([m.router.phi],lr=.03)
        if mode.endswith('TEMP'):
            for step in range(lim['outer']):
                m.zero_grad(set_to_none=True)
                J,upd,la,lb,gn=temporal_objective(m,xx[A],yy[A],xx[B],yy[B])
                gphi=torch.autograd.grad(J,m.router.phi)[0]
                m.router.phi.grad=gphi
                phign=float(nn.utils.clip_grad_norm_([m.router.phi],1))
                optphi.step()
                with torch.no_grad():
                    m.router.phi.clamp_(-2,2)
                    for n,pv in theta_items(m):pv.copy_(upd[n].detach())
                visits+=len(A)+len(B)
                row=[float(la.detach()),float(lb.detach()),float(J.detach()),float(gn.detach()),phign,float(m.router.phi.detach().norm())]
                _check_finite(row);route_hist.append(row)
        else:
            opttheta=torch.optim.SGD(theta,lr=.01)
            for step in range(lim['erm']):
                m.zero_grad(set_to_none=True)
                mse=(m(xx[pool])-yy[pool]).square().mean()
                J=mse+.001*m.router.phi.square().sum();J.backward()
                gn=float(nn.utils.clip_grad_norm_(theta,1));phign=float(nn.utils.clip_grad_norm_([m.router.phi],1))
                opttheta.step();optphi.step()
                with torch.no_grad():m.router.phi.clamp_(-2,2)
                visits+=len(pool)
                row=[float(mse.detach()),float(J.detach()),gn,phign,float(m.router.phi.detach().norm())]
                _check_finite(row);route_hist.append(row)
        phase_seconds['router_learning']=time.perf_counter()-mark
        with torch.no_grad():
            for n,pv in theta_items(m):pv.copy_(original[n])
        m.router.phi.requires_grad_(False)
    mark=time.perf_counter()
    active=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(active,lr=.003,weight_decay=.001)
    with torch.no_grad():best=float((m(xx[C])-yy[C]).square().mean())
    best_state=copy.deepcopy(m.state_dict());chosen=0;wait=0
    maxepochs=lim['final'] if is_factorial else lim['joint']
    for step in range(maxepochs):
        m.zero_grad(set_to_none=True)
        mse=(m(xx[F])-yy[F]).square().mean()
        loss=mse+(.001*m.router.phi.square().sum() if mode=='D_JOINT' else 0)
        loss.backward();gn=float(nn.utils.clip_grad_norm_(active,1));opt.step()
        if mode=='D_JOINT':
            with torch.no_grad():m.router.phi.clamp_(-2,2)
        with torch.no_grad():vl=float((m(xx[C])-yy[C]).square().mean())
        row=[float(mse.detach()),vl,gn];_check_finite(row);final_hist.append(row)
        visits+=len(F)+len(C)
        if vl<best-1e-6:
            best=vl;best_state=copy.deepcopy(m.state_dict());chosen=step+1;wait=0
        else:wait+=1
        if wait>=lim['patience']:break
    m.load_state_dict(best_state)
    phase_seconds['final_fit']=time.perf_counter()-mark
    if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise FloatingPointError('invalid weights')
    log=dict(mode=mode,seed=seed,**receipt,parameters=sum(p.numel() for p in m.parameters()),
        initial_digest=initial_digest,theta_initial_digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in original.values())).hexdigest(),
        warmup_history=warm_hist,router_history=route_hist,final_history=final_hist,selected_epoch=chosen,
        best_C_loss=best,final_epoch_count=len(final_hist),forward_sample_visits=visits,
        phase_wall_seconds=phase_seconds,wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
        final_digest=state_digest(m),router_phi=(m.router.phi.detach().tolist() if mode!='GRU' else None),
        hypergradient_order=('exact one-step; past iterates detached' if mode.endswith('TEMP') else 'ordinary first order'))
    return m,sd,log


def forecast(m,sd,p,origins):
    if len(origins)==0:return np.empty((0,20)),{}
    x=np.array([np.diff(p[t-60:t+1]) for t in origins])
    with torch.no_grad():
        xx=torch.as_tensor(x/sd,dtype=next(m.parameters()).dtype)
        pred=m(xx).cpu().numpy().astype(float)*sd
        if isinstance(m,DualGRU):
            c,d,w=m.router(xx)
            diag=dict(reconstruction_max_error=float((c+d-xx).abs().max()),route_mean=w.mean((0,1)).tolist(),
                route_std=w.flatten(0,1).std(0,unbiased=False).tolist(),route_min=float(w.min()),route_max=float(w.max()))
        else:diag={}
    if not np.isfinite(pred).all():raise FloatingPointError('nonfinite prediction')
    return pred,diag
