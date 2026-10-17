"""Causal prefix infrastructure and one bounded conditional horizon-mesh decoder.
"""
from __future__ import annotations
import csv,hashlib,io,os
from pathlib import Path
import numpy as np
import torch
from torch import nn

BASE_SHA='ba282791e1900aeed8c120535798e8532a4f5670'
PROTOCOL_SHA='2b141610106c8c5f6eb9fc138d5d23485fc682ac'
PREFIX_SHA='1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'
SEEDS=(17,29,41,53)
MODES=('MESH','GLOBAL','TILT','FIXED6','RANK4','RANK6','DIRECT','GRU')
H=20; GAP_FLOOR=.25
BANK=[(0.,1.)]+[(r,q) for r in (0.,.25,.5,.75,.9,.97) for q in (.02,.1,.25,.5,.8)]

def configure():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False

def load_prefix(path):
    with Path(path).open('rb',buffering=0) as f: raw=f.read(29204)
    if len(raw)!=29204 or hashlib.sha256(raw).hexdigest()!=PREFIX_SHA:
        raise ValueError('Approved prefix bytes differ; no fallback or suffix read')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    dates=np.array([a['date'] for a in rows]);p=np.array([float(a['author_reported_price']) for a in rows])
    if len(p)!=950 or dates[-1]!='2021-12-31' or not np.all(dates[1:]>dates[:-1]):raise ValueError('date support')
    if not np.isfinite(p).all() or np.any(p<=0):raise ValueError('price invalid')
    return p,dates,dict(records=950,bytes=len(raw),last_date=str(dates[-1]),prefix_sha256=PREFIX_SHA,next_byte_requested=False)

def reference(p,window=240):
    """Inherited causal 31-member baseline, re-estimated from new price input."""
    rho,q=np.array(BANK).T;r=1-q;n=len(p);z=np.zeros(len(q));P=r/(1-rho*rho)
    zz=np.zeros((n,len(q)));ee=np.zeros_like(zz);ss=np.ones_like(zz)
    for t in range(1,n):
        e=p[t]-p[t-1]-(rho-1)*z;S=(rho-1)**2*P+1.;C=rho*(rho-1)*P+r
        z=rho*z+C*e/S;P=np.maximum(rho*rho*P+r-C*C/S,0.)
        zz[t],ee[t],ss[t]=z,e,S
    out={k:np.zeros(n) for k in ('mu','rho','q')};out['b']=np.zeros((n,H));out['selected']=np.zeros(n,dtype=np.int64)
    j=0
    for t in range(n):
        if t>=60 and (t-60)%20==0:
            start=max(1,t-window);v=np.maximum(np.mean(ee[start:t]**2/ss[start:t],0),1e-12)
            j=int(np.argmin(np.log(ss[start:t]).sum(0)+(t-start)*np.log(v)))
        rr,vv=BANK[j];out['selected'][t]=j;out['mu'][t]=zz[t,j];out['rho'][t]=rr;out['q'][t]=vv
        out['b'][t]=(rr**np.arange(1,H+1)-1)*zz[t,j]
    return out

def support(p,end):
    ids=np.arange(max(60,end-H-480),end-H);split=len(ids)-60
    tr=np.arange(split-20);va=np.arange(split,len(ids))
    if len(tr)<60 or ids[tr[-1]]+H>=ids[va[0]] or ids[-1]+H>=end:raise ValueError('maturity/purge')
    x=np.stack([np.diff(p[t-60:t+1]) for t in ids]);y=np.stack([p[t+1:t+H+1]-p[t] for t in ids])
    sd=max(float(x[tr].std()),1e-6)
    return ids,x,y,tr,va,sd


def context(ref,ids,sd):
    return np.column_stack([np.tanh(ref['mu'][ids]/sd),
        ref['rho'][ids]*(1-ref['q'][ids]),1-ref['q'][ids]])

def batch(p,ref,ids,sd):
    ids=np.asarray(ids,dtype=np.int64)
    x=np.stack([np.diff(p[i-60:i+1]) for i in ids])/sd
    return tuple(torch.as_tensor(a,dtype=torch.float64) for a in
        (x,context(ref,ids,sd),ref['b'][ids]/sd))



def initial_basis(rank):
    """Inherited fixed-knot control; not an N-HiTS implementation."""
    knots=np.rint(np.geomspace(1,H,rank)).astype(int)
    if len(np.unique(knots))!=rank:raise ValueError('duplicate knots')
    points=np.r_[0,knots]
    out=np.stack([np.interp(np.arange(1,H+1),points,np.r_[0,np.eye(rank)[i]])
                  for i in range(rank)],axis=1)
    return torch.as_tensor(out,dtype=torch.get_default_dtype())


def knots_from_logits(z):
    """Five gaps > .25. Absolute endpoints 1 and 20 are fixed."""
    g0=z.new_tensor([1.,1.,3.,5.,9.])
    total=19.-5*GAP_FLOOR
    log_prior=torch.log((g0-GAP_FLOOR)/total)
    gaps=GAP_FLOOR+total*torch.softmax(log_prior+z,dim=-1)
    first=torch.ones_like(gaps[:,:1])
    return torch.cat([first,first+torch.cumsum(gaps,dim=-1)],dim=-1)


def interpolate(knots,values,query=None):
    """Continuous piecewise-linear response, including the fixed origin (0,0).
    The ramp form avoids nondifferentiable sorting; derivatives exist almost
    everywhere, with conventional subgradients at the finite knot set.
    """
    if query is None:query=torch.arange(1,H+1,dtype=knots.dtype,device=knots.device)
    query=torch.as_tensor(query,dtype=knots.dtype,device=knots.device)
    points=torch.cat([knots.new_zeros((knots.shape[0],1)),knots],dim=-1)
    vals=torch.cat([values.new_zeros((values.shape[0],1)),values],dim=-1)
    gap=points[:,1:]-points[:,:-1]
    ramp=((query[None,:,None]-points[:,None,:-1])/gap[:,None,:]).clamp(0.,1.)
    return torch.sum(ramp*(vals[:,1:]-vals[:,:-1])[:,None,:],dim=-1)


class Forecast(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.encoder=nn.GRU(1,16,batch_first=True)
        if mode=='GRU':self.head=nn.Linear(16,H);return
        self.U=nn.Linear(16,8,bias=False);self.V=nn.Linear(3,8,bias=False)
        self.bias=nn.Parameter(torch.zeros(8))
        if mode=='DIRECT':
            self.head=nn.Linear(8,H,bias=False);nn.init.zeros_(self.head.weight);return
        rank=4 if mode=='RANK4' else 6
        self.coefficients=nn.Linear(8,rank)
        nn.init.zeros_(self.coefficients.weight);nn.init.zeros_(self.coefficients.bias)
        B=initial_basis(rank)
        if mode in ('RANK4','RANK6'):self.basis=nn.Parameter(B)
        else:self.register_buffer('basis',B)
        self.drift=nn.Linear(8,1)
        nn.init.zeros_(self.drift.weight);nn.init.zeros_(self.drift.bias)
        # Construct geometry after all common modules: same seed gives identical
        # common parameters and initial predictions in matched spline controls.
        if mode=='MESH':
            self.geometry=nn.Linear(8,4);nn.init.zeros_(self.geometry.weight);nn.init.zeros_(self.geometry.bias)
        elif mode=='GLOBAL':self.geometry=nn.Parameter(torch.zeros(4))
        elif mode=='TILT':
            self.geometry=nn.Linear(8,1);nn.init.zeros_(self.geometry.weight);nn.init.zeros_(self.geometry.bias)

    def features(self,x,q):
        h=self.encoder(x[:,:,None])[0][:,-1]
        hz=self.encoder(torch.zeros((1,60,1),dtype=x.dtype,device=x.device))[0][0,-1]
        return torch.tanh(self.U(h-hz)+self.V(q)+self.bias)-torch.tanh(self.bias)

    def knot_locations(self,f):
        n=len(f)
        if self.mode=='MESH':z=torch.cat([2*torch.tanh(self.geometry(f)),f.new_zeros((n,1))],dim=-1)
        elif self.mode=='GLOBAL':z=torch.cat([2*torch.tanh(self.geometry)[None].expand(n,-1),f.new_zeros((n,1))],dim=-1)
        elif self.mode=='TILT':z=2*torch.tanh(self.geometry(f))*f.new_tensor([-1.,-.5,0.,.5,1.])[None]
        else:return f.new_tensor([1.,2.,3.,6.,11.,20.])[None].expand(n,-1)
        return knots_from_logits(z)

    def forward(self,x,q,base,details=False):
        if self.mode=='GRU':
            pred=self.head(self.encoder(x[:,:,None])[0][:,-1])
            return (pred,dict(correction=pred)) if details else pred
        f=self.features(x,q)
        if self.mode=='DIRECT':correction=self.head(f);info={}
        else:
            coeff=3*torch.tanh(self.coefficients(f));d=torch.tanh(self.drift(f)).squeeze(-1)
            grid=torch.arange(1,H+1,dtype=x.dtype,device=x.device)/H
            if self.mode in ('MESH','GLOBAL','TILT'):
                knots=self.knot_locations(f);curve=interpolate(knots,coeff);info=dict(knots=knots)
            else:curve=coeff@self.basis.T;info={}
            correction=curve+d[:,None]*grid
            info.update(coefficients=coeff,drift=d)
        info['correction']=correction
        return (base+correction,info) if details else base+correction


def make_model(mode,seed):
    configure();torch.manual_seed(int(seed));return Forecast(mode)
