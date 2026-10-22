"""Purged forward-risk response-subspace learner and matched ordinary controls."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import legacy

SEEDS=(17,29,41,53)
MODES=('GAIN','FWD_FIXED','IN_SAMPLE','VARPRO_FIXED','FROZEN_GAIN','FIXED6','RANK6','GRU')
PROFILE_MODES=MODES[:5]
H=20; RIDGE=.01; MAX_RANK=6; POSITIVE_TOL=1e-10
PREFIX_SHA=legacy.PREFIX_SHA
configure=legacy.configure
load_prefix=legacy.load_prefix
reference=legacy.reference
context=legacy.context
batch=legacy.batch
initial_basis=legacy.initial_basis


def folds(n):
    pairs=[]
    for fraction in (.60,.75):
        cut=int(np.floor(fraction*n))
        a=np.arange(cut); b=np.arange(cut+20,n)
        if len(a)<60 or len(b)<20: raise ValueError('insufficient purged support')
        pairs.append((a,b))
    return pairs


def support(p,end):
    ids,x,y,tr,va,_=legacy.support(p,end)
    a=folds(len(tr))[0][0]
    sd=max(float(x[tr[a]].std()),1e-6)
    return ids,x,y,tr,va,sd


def fixed_projector(dtype=None,device=None):
    b=initial_basis(6).to(dtype=dtype,device=device)
    q=torch.linalg.qr(b,mode='reduced').Q
    return q@q.T


def ridge(z,r):
    n=len(z)
    eye=torch.eye(z.shape[1],dtype=z.dtype,device=z.device)
    return torch.linalg.solve(2*(z.T@z)/n+RIDGE*eye,(z.T@r)/n)


def gain_matrix(z,r,a,b):
    d=ridge(z[a],r[a]); u=z[b]@d
    s=(r[b].T@u+u.T@r[b]-2*u.T@u)/len(b)-RIDGE*(d.T@d)
    return (s+s.T)/2, d


def spectral(s):
    values,vectors=torch.linalg.eigh(s)
    take=torch.zeros_like(values,dtype=torch.bool)
    take[-MAX_RANK:]=values[-MAX_RANK:]>POSITIVE_TOL
    q=vectors[:,take]
    return q@q.T,values,take


def system(z,r,mode,need_projector=True):
    """Returns actual finite-pair risk, without differentiating eigenvectors."""
    if mode in ('IN_SAMPLE','VARPRO_FIXED'):
        a=np.arange(len(z));pairs=[(a,a)]
    else:pairs=folds(len(z))
    matrices=[]; constants=[]
    for a,b in pairs:
        s,_=gain_matrix(z,r,a,b);matrices.append(s);constants.append(r[b].square().sum()/len(b))
    s=torch.stack(matrices).mean(0)
    if mode in ('FWD_FIXED','VARPRO_FIXED'):
        p=fixed_projector(z.dtype,z.device); value=torch.trace(p@s)
        eig=torch.linalg.eigvalsh(s.detach());rank=6
    else:
        eig=torch.linalg.eigvalsh(s)
        top=eig[-MAX_RANK:];value=top[top>POSITIVE_TOL].sum()
        rank=int((top>POSITIVE_TOL).sum())
        p=spectral(s.detach())[0] if need_projector else None
    loss=(torch.stack(constants).mean()-value)/H
    return loss,p,dict(eigenvalues=eig.detach(),rank=rank,gain=s.detach())


class ProfileNet(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.encoder=nn.GRU(1,16,batch_first=True)
        self.U=nn.Linear(16,8,bias=False);self.V=nn.Linear(3,8,bias=False)
        self.bias=nn.Parameter(torch.zeros(8))
        self.register_buffer('D',torch.zeros(9,H));self.register_buffer('P',torch.zeros(H,H))

    def features(self,x,q):
        h=self.encoder(x[:,:,None])[0][:,-1]
        hz=self.encoder(torch.zeros((1,60,1),dtype=x.dtype,device=x.device))[0][0,-1]
        f=torch.tanh(self.U(h-hz)+self.V(q)+self.bias)-torch.tanh(self.bias)
        return torch.cat([f,torch.ones((len(f),1),dtype=f.dtype,device=f.device)],dim=1)

    def forward(self,x,q,base,details=False):
        z=self.features(x,q);correction=z@self.D@self.P
        info=dict(correction=correction)
        return (base+correction,info) if details else base+correction

    def install(self,x,q,residual):
        with torch.no_grad():
            z=self.features(x,q);_,p,info=system(z,residual,self.mode)
            self.D.copy_(ridge(z,residual));self.P.copy_(p)
            info['condition']=float(torch.linalg.cond(2*z.T@z/len(z)+RIDGE*torch.eye(9)))
        return info


def make_model(mode,seed):
    configure();torch.manual_seed(int(seed))
    return ProfileNet(mode) if mode in PROFILE_MODES else legacy.Forecast(mode)
