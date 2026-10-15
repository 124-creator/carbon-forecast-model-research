"""Causal-moment recurrent learning, not a new generic GLS/GMM theorem.

No network access or implicit file writes. All training-dependent state is
created by the caller from the approved data prefix. Nonsymmetric moment
updates deliberately do not differentiate through the analytic head.
"""
from __future__ import annotations
import csv, hashlib, io, functools
from pathlib import Path
import numpy as np
from scipy.linalg import solveh_banded
import torch
from torch import nn

PREFIX_BYTES=29204
PREFIX_SHA='1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'
BASE_SHA='13e7a6bf129916df5807746664b30df1c1936e73'
PROTOCOL_SHA='b31a5198725c2015cd22d06916c4bc2c4311f6f5'
H=20;ALPHA=.01;GAMMA=1.
SEEDS=(17,29,41,53)
MODES=('CAUSAL','IDENTITY','SYMMETRIC','SHUFFLED','FROZEN','FULL_WINDOW','GRU_WINDOW')
BANK=[(0.,1.)]+[(r,q) for r in (0.,.25,.5,.75,.9,.97) for q in (.02,.1,.25,.5,.8)]

def configure():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)

def load_prefix(path):
    with Path(path).open('rb',buffering=0) as f:raw=f.read(PREFIX_BYTES)
    if len(raw)!=PREFIX_BYTES or hashlib.sha256(raw).hexdigest()!=PREFIX_SHA:
        raise ValueError('Approved prefix differs; no suffix/fallback is permitted')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    p=np.array([float(r['author_reported_price']) for r in rows]);dates=np.array([r['date'] for r in rows])
    if len(p)!=950 or dates[-1]!='2021-12-31' or not np.all(dates[1:]>dates[:-1]):raise ValueError('date contract')
    if not np.isfinite(p).all() or not np.all(p>0):raise ValueError('invalid prices')
    return p,dates,raw

def reference(p,window=240):
    """Existing causal reference; recomputed, never loaded from previous fits."""
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

@functools.lru_cache(maxsize=32)
def precision(n,h):
    """Inverse of .5I+.5 interval Gram, unit trace/n; data independent."""
    band=np.zeros((h,n));band[0]=1.
    for k in range(1,h):band[k,:n-k]=.5*(h-k)/h
    out=solveh_banded(band,np.eye(n),lower=True,check_finite=True)
    out=(out+out.T)/2;out*=n/np.trace(out);out.flags.writeable=False
    return out

@functools.lru_cache(maxsize=32)
def operator_array(n,mode):
    matrices=[]
    for h in range(1,H+1):
        if mode=='IDENTITY':u=np.eye(n)
        else:
            p=precision(n,h).copy()
            if mode=='SHUFFLED':
                perm=np.random.default_rng(202615+h).permutation(n);p=p[np.ix_(perm,perm)]
            u=p if mode=='SYMMETRIC' else np.diag(np.diag(p))+2*np.triu(p,1)
        matrices.append(u)
    return np.array(matrices)

def solve_head(g,target,operator,alpha=ALPHA,gamma=GAMMA):
    """Regularized moment equation; positive symmetric part ensures unique head."""
    n,d=g.shape
    ug=torch.matmul(operator,g)
    a=torch.matmul(g.T.unsqueeze(0),ug)/n+gamma*(g.T@g).unsqueeze(0)/n+alpha*torch.eye(d,dtype=g.dtype)
    rhs=torch.matmul(g.T.unsqueeze(0),torch.matmul(operator,target.T.unsqueeze(-1)))/n
    return torch.linalg.solve(a,rhs).squeeze(-1)

def score_direction(pred,target,operator,gamma=GAMMA):
    return torch.matmul(operator,(pred-target).T.unsqueeze(-1)).squeeze(-1).T+gamma*pred

class Feature(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.GRU(1,16,batch_first=True)
        self.U=nn.Linear(16,8,bias=False);self.V=nn.Linear(3,8,bias=False);self.c=nn.Parameter(torch.zeros(8))
    def forward(self,x,q,indices=None):
        h=self.encoder(x)[0]
        h=h[0,indices] if indices is not None else h[:,-1]
        return torch.tanh(self.U(h)+self.V(q)+self.c)-torch.tanh(self.c)

class Model(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        if mode=='GRU_WINDOW':
            self.encoder=nn.GRU(1,16,batch_first=True);self.head=nn.Linear(16,H)
        else:
            self.feature=Feature();self.B=nn.Parameter(torch.zeros(H,8))
    def encode(self,batch):
        return self.feature(batch['x'],batch['q'],batch.get('indices'))
    def forward(self,batch):
        if self.mode=='GRU_WINDOW':return self.head(self.encoder(batch['x'])[0][:,-1])
        return batch['base']+self.encode(batch)@self.B.T

def make_model(mode,seed):
    if mode not in MODES:raise ValueError(mode)
    configure();torch.manual_seed(seed);return Model(mode)

def batch(p,ref,orig,sd,mode):
    orig=np.asarray(orig,dtype=np.int64)
    if len(orig)==0 or orig.min()<1 or orig.max()>=len(p):raise ValueError('origins')
    q=np.column_stack([np.tanh(ref['mu'][orig]/sd),ref['rho'][orig]*(1-ref['q'][orig]),1-ref['q'][orig]])
    tensor=lambda a:torch.as_tensor(np.array(a,copy=True),dtype=torch.float64)
    out={'q':tensor(q),'base':tensor(ref['b'][orig]/sd)}
    if mode.endswith('WINDOW'):
        out['x']=tensor(np.stack([np.diff(p[t-60:t+1]) for t in orig])/sd)[:,:,None]
    else:
        # RNN output index t-1 corresponds exactly to price observation t.
        out['x']=tensor(np.diff(p[:orig.max()+1])/sd)[None,:,None]
        out['indices']=torch.as_tensor(orig-1,dtype=torch.long)
    return out

def predict(m,p,ref,orig,sd):
    with torch.no_grad():return m(batch(p,ref,orig,sd,m.mode)).numpy()*sd
