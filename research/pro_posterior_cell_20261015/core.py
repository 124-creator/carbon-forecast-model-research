"""Stage J: one conditional-moment recurrent architecture and matched controls.

The reference is a working Gaussian model. Its variance is not observed price
noise. Only the one-step sine proposal is analytically integrated; the full
recurrent trajectory is NOT integrated as a Bayesian posterior.
"""
from __future__ import annotations
import csv, hashlib, io, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PREFIX_HASH = '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'
BASE_SHA = 'da05a07f6d8874a7192b772b4ddbb00d765da17e'
SEEDS = (17,29,41,53)
MODES = ('INTEGRATED','POINT','FREE_DAMP','CONSTANT','FULL','GRU')
HORIZONS = (1,5,20)
BANK = [(0.,1.)]+[(r,q) for r in (0.,.25,.5,.75,.9,.97) for q in (.02,.1,.25,.5,.8)]

def load_prefix(path):
    """Read exactly the approved byte count; no whole-file hash or next row."""
    with Path(path).open('rb', buffering=0) as f:
        raw=f.read(29204)
    if len(raw)!=29204 or hashlib.sha256(raw).hexdigest()!=PREFIX_HASH:
        raise ValueError('Approved prefix mismatch; refuse normalization or expansion')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    dates=np.array([r['date'] for r in rows]); p=np.array([float(r['author_reported_price']) for r in rows])
    if len(p)!=950 or dates[-1]!='2021-12-31' or not np.all(dates[1:]>dates[:-1]):
        raise ValueError('Invalid date support')
    if not np.isfinite(p).all() or not np.all(p>0): raise ValueError('Invalid prices')
    return p, dates, {'records':950,'bytes':29204,'last_date':dates[-1],
                     'sha256':PREFIX_HASH,'next_byte_requested':False}

def filter_bank(p):
    """Causal Gaussian filtering with conditional, normalized state variance."""
    n=len(p); rho,q=np.array(BANK).T; r=1-q
    z=np.zeros(len(BANK)); P=r/(1-rho*rho)
    zz=np.zeros((n,len(BANK))); pp=np.zeros_like(zz); ee=np.zeros_like(zz); ss=np.ones_like(zz)
    pp[0]=P
    for t in range(1,n):
        e=p[t]-p[t-1]-(rho-1)*z; S=(rho-1)**2*P+1
        C=rho*(rho-1)*P+r
        z=rho*z+C*e/S; P=np.maximum(rho*rho*P+r-C*C/S,0.)
        zz[t],pp[t],ee[t],ss[t]=z,P,e,S
    return zz,pp,ee,ss

def reference(p, window=240):
    zz,pp,ee,ss=filter_bank(p); n=len(p)
    out={k:np.zeros(n) for k in ('mu','var','rho','q','scale')}
    out['b']=np.zeros((n,20)); out['selected']=np.zeros(n,dtype=np.int64)
    j=0; scale=1.
    for t in range(n):
        if t>=60 and (t-60)%20==0:
            start=max(1,t-window)
            scales=np.maximum(np.mean(ee[start:t]**2/ss[start:t],axis=0),1e-12)
            obj=np.log(ss[start:t]).sum(0)+(t-start)*np.log(scales)
            j=int(np.argmin(obj)); scale=float(scales[j])
        rho,q=BANK[j]; out['selected'][t]=j
        out['mu'][t]=zz[t,j]; out['var'][t]=scale*pp[t,j]
        out['rho'][t]=rho; out['q'][t]=q; out['scale'][t]=scale
        out['b'][t]=(rho**np.arange(1,21)-1)*zz[t,j]
    return out

def support(p,end):
    ids=np.arange(max(60,end-20-480),end-20)
    split=len(ids)-60; tr=np.arange(split-20); va=np.arange(split,len(ids))
    if len(tr)<60 or ids[tr[-1]]+20>=ids[va[0]] or ids[-1]+20>=end:
        raise ValueError('Maturity/purge violation')
    x=np.array([np.diff(p[t-60:t+1]) for t in ids])
    y=np.array([p[t+1:t+21]-p[t] for t in ids])
    sd=max(float(x[tr].std()),1e-6)
    return ids,x,y,tr,va,sd

def features(p,ref,ids,sd):
    ids=np.asarray(ids,dtype=int)
    x=np.array([np.diff(p[t-60:t+1]) for t in ids])/sd
    mu=np.array([ref['mu'][t-59:t+1] for t in ids])/sd
    var=np.array([ref['var'][t-59:t+1] for t in ids])/(sd*sd)
    q=np.column_stack([np.tanh(ref['mu'][ids]/sd),ref['rho'][ids]*(1-ref['q'][ids]),1-ref['q'][ids]])
    return tuple(torch.as_tensor(a,dtype=torch.float64) for a in (x,mu,var,q,ref['b'][ids]/sd))

@torch.jit.script
def moment_scan(gx:torch.Tensor, mu:torch.Tensor, var:torch.Tensor,
                wh:torch.Tensor, bh:torch.Tensor, a:torch.Tensor,
                damp:torch.Tensor, mode:int, vbar:torch.Tensor):
    width=a.shape[0]
    h=torch.zeros((gx.shape[0],width),dtype=gx.dtype,device=gx.device)
    for t in range(gx.shape[1]):
        gh=torch.nn.functional.linear(h,wh,bh)
        rt=torch.sigmoid(gx[:,t,:width]+gh[:,:width])
        zt=torch.sigmoid(gx[:,t,width:2*width]+gh[:,width:2*width])
        eta=gx[:,t,2*width:]+rt*gh[:,2*width:]
        phase=eta+a*mu[:,t:t+1]
        psi=torch.sin(phase)
        if mode==1:
            psi=psi*torch.exp(-.5*a.square()*var[:,t:t+1])
        elif mode==2:
            psi=psi*torch.exp(-.5*damp.square()*var[:,t:t+1])
        elif mode==3:
            psi=psi*torch.exp(-.5*a.square()*vbar)
        h=zt*h+(1-zt)*psi
    return h

class MomentGRU(nn.Module):
    """Moment tie is the only proposed mechanism; other modes are controls."""
    def __init__(self, mode, constant_var=0.):
        super().__init__()
        if mode not in MODES[:4]: raise ValueError(mode)
        self.mode=mode; self.code={'POINT':0,'INTEGRATED':1,'FREE_DAMP':2,'CONSTANT':3}[mode]
        self.cell=nn.GRUCell(1,16,dtype=torch.float64)
        self.a=nn.Parameter(torch.full((16,),.5,dtype=torch.float64))
        if mode=='FREE_DAMP': self.damp=nn.Parameter(self.a.detach().clone())
        else: self.register_buffer('damp',torch.zeros(16,dtype=torch.float64))
        self.register_buffer('constant_var',torch.tensor(float(constant_var),dtype=torch.float64))
        self.head=nn.Linear(16,20,dtype=torch.float64)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def encode(self,x,mu,var):
        gx=F.linear(x[:,:,None],self.cell.weight_ih,self.cell.bias_ih)
        return moment_scan(gx,mu,var,self.cell.weight_hh,self.cell.bias_hh,
                           self.a,self.damp,self.code,self.constant_var)
    def forward(self,x,mu,var,q,b):
        c=self.head(self.encode(x,mu,var)); return b+c,c
    @torch.no_grad()
    def project(self):
        self.a.clamp_(-3,3)
        if self.mode=='FREE_DAMP': self.damp.clamp_(-3,3)

class FullGRU(nn.Module):
    """Conventional Stage I FULL, freshly fitted in current float64 environment."""
    def __init__(self):
        super().__init__(); self.encoder=nn.GRU(1,16,batch_first=True,dtype=torch.float64)
        self.U=nn.Linear(16,8,bias=False,dtype=torch.float64)
        self.V=nn.Linear(3,8,bias=False,dtype=torch.float64)
        self.bias=nn.Parameter(torch.zeros(8,dtype=torch.float64))
        self.O=nn.Linear(8,20,bias=False,dtype=torch.float64); nn.init.zeros_(self.O.weight)
    def forward(self,x,mu,var,q,b):
        h=self.encoder(x[:,:,None])[0][:,-1]
        zero=self.encoder(torch.zeros((1,x.shape[1],1),dtype=x.dtype))[0][:,-1]
        c=self.O(torch.tanh(self.U(h-zero)+self.V(q)+self.bias)-torch.tanh(self.bias))
        return b+c,c
    def project(self): pass

class PlainGRU(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=nn.GRU(1,16,batch_first=True,dtype=torch.float64)
        self.head=nn.Linear(16,20,dtype=torch.float64)
    def forward(self,x,mu,var,q,b):
        c=self.head(self.encoder(x[:,:,None])[0][:,-1]); return c,c
    def project(self): pass

def make_model(mode,seed,constant_var=0.):
    torch.manual_seed(seed)
    if mode=='FULL':return FullGRU()
    if mode=='GRU':return PlainGRU()
    return MomentGRU(mode,constant_var)

def forecast(model,sd,p,ref,ids):
    with torch.no_grad():
        pred,corr=model(*features(p,ref,ids,sd))
        if isinstance(model,PlainGRU): return pred.numpy()*sd
        return ref['b'][np.asarray(ids)]+corr.numpy()*sd

def configure():
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled=False
