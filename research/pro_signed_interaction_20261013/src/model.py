"""Direct, sign-equivariant interaction predictor. No external I/O or latent states.

Rows of the encoder are learned; the ridge output layer is profiled exactly.
All covariance/scale estimates are training-only. New Stage F implementation.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from scipy.linalg import solve
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
H = np.array([1, 5, 20], dtype=int)
BLOCKS = (1, 1, 1, 2, 5, 10, 20, 20)
Q = np.zeros((8,60))
a=0
for j,b in enumerate(BLOCKS):
    Q[j,a:a+b]=1/np.sqrt(b);a+=b

def prepare(p):
    p=np.asarray(p,float)
    if p.ndim!=1 or not np.isfinite(p).all(): raise ValueError('Finite one-dimensional prices required')
    n=len(p); raw=np.zeros((n,60));x=np.zeros((n,8));s=np.zeros(n)
    for t in range(60,n):
        raw[t]=np.diff(p[t-60:t+1])[::-1]
        s[t]=np.sqrt(np.mean(raw[t]**2))
        if s[t]>0:x[t]=Q@raw[t]/s[t]
    return x,s,raw

def targets(p,idx):
    idx=np.asarray(idx,int)
    if len(idx)==0 or idx.min()<0 or idx.max()+20>=len(p):raise ValueError('Targets out of bounds')
    return np.asarray(p)[idx[:,None]+H]-np.asarray(p)[idx,None]

def training_scales(p,n_initial):
    p=np.asarray(p)[:n_initial]
    ss=np.array([np.mean((p[h:]-p[:-h])**2) for h in H])
    if not np.isfinite(ss).all() or (ss<=0).any():raise ValueError('Degenerate training scale')
    return np.sqrt(ss)

def support(fit_end,window=None):
    # fit_end is exclusive; max target <= fit_end-1.
    end=int(fit_end)-20
    start=max(60,end-window) if window is not None else 60
    if end-start<40:raise ValueError('Insufficient complete mature labels')
    return np.arange(start,end)

class Encoder(torch.nn.Module):
    def __init__(self,kind,seed):
        super().__init__();self.kind=kind
        gen=torch.Generator().manual_seed(int(seed))
        if kind in ('INTERACTION','FROZEN'):
            self.theta=torch.nn.Parameter(torch.randn((3,4,8),generator=gen)*.7/np.sqrt(8))
        elif kind=='ODD_MLP':
            self.theta=torch.nn.Parameter(torch.randn((10,8),generator=gen)*.7/np.sqrt(8))
        elif kind=='MLP':
            self.theta=torch.nn.Parameter(torch.randn((9,9),generator=gen)*.7/np.sqrt(9))
        elif kind in ('LINEAR8','LINEAR60'):
            self.theta=torch.nn.Parameter(torch.zeros(0),requires_grad=False)
        else:raise ValueError(kind)
        self.project()
    def forward(self,x):
        if self.kind in ('INTERACTION','FROZEN'):
            u,v,w=[x@r.T for r in self.theta]
            return torch.cat([x,u*torch.tanh(v*w)],dim=-1)
        if self.kind=='ODD_MLP':return torch.cat([x,torch.tanh(x@self.theta.T)],dim=-1)
        if self.kind=='MLP':
            one=torch.ones((len(x),1),dtype=x.dtype,device=x.device)
            return torch.cat([x,torch.tanh(torch.cat([x,one],1)@self.theta.T),one],1)
        return x
    def project(self):
        with torch.no_grad():
            if self.theta.numel():
                self.theta.div_(self.theta.norm(dim=-1,keepdim=True).clamp_min(1))
    def count(self):
        m={'INTERACTION':12,'FROZEN':12,'MLP':18,'ODD_MLP':18,'LINEAR8':8,'LINEAR60':60}[self.kind]
        return int(self.theta.numel()+m*3)

def profiled(enc,x,loc,y,hs,alpha):
    f=enc(x);ratio=loc[:,None]/hs[None,:]
    A=f[None,:,:]*ratio.T[:,:,None]
    yy=(y/hs).T[:,:,None]
    n=len(x);d=f.shape[1]
    gram=A.transpose(1,2)@A/n+alpha*torch.eye(d)[None,:,:]
    B=torch.linalg.solve(gram,A.transpose(1,2)@yy/n)
    pred=A@B
    loss=((pred-yy)**2).mean()+alpha*(B**2).sum()/3
    if enc.theta.numel():loss=loss+1e-4*(enc.theta**2).sum()
    return loss,B.squeeze(-1).T

class ProfilePredictor:
    def __init__(self,enc,B,hs,log):self.enc=enc;self.B=B;self.hs=hs;self.log=log
    def predict(self,x,local):
        with torch.no_grad():
            f=self.enc(torch.as_tensor(x))
            out=torch.as_tensor(local)[:,None]*(f@self.B)
        if not torch.isfinite(out).all():raise FloatingPointError('Nonfinite prediction')
        return out.numpy()

def fit_profile(kind,x,loc,y,hs,alpha,seed,steps=160):
    begin=time.monotonic();enc=Encoder(kind,seed)
    xt=torch.as_tensor(np.asarray(x));lt=torch.as_tensor(np.asarray(loc));yt=torch.as_tensor(np.asarray(y));st=torch.as_tensor(np.asarray(hs))
    trained=kind in ('INTERACTION','MLP','ODD_MLP')
    opt=torch.optim.Adam(enc.parameters(),lr=.02) if trained else None
    best=float('inf');best_state=None;hist=[]
    for it in range((steps if trained else 0)+1):
        if opt:opt.zero_grad()
        loss,B=profiled(enc,xt,lt,yt,st,float(alpha))
        value=float(loss.detach())
        if not np.isfinite(value):raise FloatingPointError('Nonfinite profiled objective')
        if value<best:best=value;best_state=enc.theta.detach().clone()
        if it%40==0 or it==steps:hist.append([it,value])
        if trained and it<steps:
            loss.backward();torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0);opt.step();enc.project()
    with torch.no_grad():enc.theta.copy_(best_state)
    with torch.no_grad():final,B=profiled(enc,xt,lt,yt,st,float(alpha))
    log=dict(kind=kind,seed=seed,alpha=float(alpha),n_train=len(x),steps=steps if trained else 0,
             parameters=enc.count(),train_objective=float(final),history=hist,seconds=time.monotonic()-begin)
    return ProfilePredictor(enc,B.detach(),hs,log)

class KernelPredictor:
    def __init__(self,x,loc,coef,hs,band,odd):self.x=x;self.loc=loc;self.coef=coef;self.hs=hs;self.band=band;self.odd=odd
    def predict(self,x,local):
        k=kernel(x,self.x,self.band,self.odd)
        ans=np.empty((len(x),3))
        for j,sh in enumerate(self.hs):ans[:,j]=np.asarray(local)*(k@((self.loc/sh)*self.coef[j]))
        return ans

def kernel(x,z,band,odd=False):
    x=np.asarray(x);z=np.asarray(z)
    xx=(x*x).sum(1)[:,None];zz=(z*z).sum(1)[None,:];dot=x@z.T
    k=np.exp(-np.maximum(xx+zz-2*dot,0)/(2*band*band))
    if odd:return .5*(k-np.exp(-np.maximum(xx+zz+2*dot,0)/(2*band*band)))
    return k+1.0 # a penalized constant feature, not an unpenalized intercept

def fit_kernel(x,loc,y,hs,alpha,band,odd):
    n=len(x);k=kernel(x,x,band,odd);coef=[]
    for j,sh in enumerate(hs):
        a=loc/sh;G=a[:,None]*k*a[None,:]+float(alpha)*n*np.eye(n)
        coef.append(solve(G,y[:,j]/sh,assume_a='pos',check_finite=True))
    return KernelPredictor(x,np.asarray(loc),coef,np.asarray(hs),band,odd)

class TreePredictor:
    def __init__(self,models):self.models=models
    def predict(self,x,local):return np.asarray(local)[:,None]*np.stack([m.predict(x) for m in self.models],1)

def fit_tree(x,loc,y,hs,leaves,lr,l2):
    # Predict normalized changes, weighted to equal raw price-MSE objective.
    active=np.asarray(loc)>0;x=np.asarray(x)[active];yy=np.asarray(y)[active]/np.asarray(loc)[active,None]
    wt=np.asarray(loc)[active]**2;wt=wt/wt.mean();models=[]
    for j in range(3):
        m=HistGradientBoostingRegressor(max_leaf_nodes=leaves,learning_rate=lr,l2_regularization=l2,
            max_iter=80,early_stopping=False,random_state=0,min_samples_leaf=20)
        m.fit(x,yy[:,j],sample_weight=wt);models.append(m)
    return TreePredictor(models)
