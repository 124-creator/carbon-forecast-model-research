"""Reference-anchored recurrent mixed differences; no network or file side effects.

Anchored differences remove main effects at specified zero anchors. They do NOT
establish population orthogonality or economic-state identifiability.
"""
from __future__ import annotations
import csv, hashlib, io
from pathlib import Path
import numpy as np
import torch
from torch import nn

PREFIX_HASH = '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'
SEEDS = (17, 29, 41, 53)
HORIZONS = (1, 5, 20)
MODES = ('RESIDUAL', 'ADDITIVE', 'FULL', 'MIXED')
LAMBDAS = (0., 1., 10.)
BANK = [(0., 1.)] + [(r, q) for r in (0., .25, .5, .75, .9, .97)
                                for q in (.02, .1, .25, .5, .8)]

def load_prefix(path: str | Path):
    """Unbuffered exact prefix, never asks for the next row or byte."""
    parts = []
    with Path(path).open('rb', buffering=0) as f:
        for _ in range(951):
            line = bytearray()
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError('Truncated or unterminated approved prefix')
                line.extend(c)
                if c == b'\n':
                    break
            parts.append(bytes(line))
    raw = b''.join(parts)
    if len(raw) != 29204 or hashlib.sha256(raw).hexdigest() != PREFIX_HASH:
        raise ValueError('Approved prefix hash mismatch; do not silently normalize')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    dates = np.array([r['date'] for r in rows])
    prices = np.array([float(r['author_reported_price']) for r in rows])
    if len(prices) != 950 or dates[-1] != '2021-12-31':
        raise ValueError('Unapproved period')
    if not np.all(np.isfinite(prices)) or not np.all(prices > 0):
        raise ValueError('Invalid target prices')
    if not np.all(dates[1:] > dates[:-1]):
        raise ValueError('Unsorted/duplicate dates')
    return prices, dates, dict(bytes=len(raw), rows=950, last_date=dates[-1],
                              prefix_sha256=PREFIX_HASH, next_byte_requested=False)

def filter_bank(p: np.ndarray):
    """31 normalized local-level/transient Gaussian observers, price only."""
    n, k = len(p), len(BANK)
    rho, q = np.array(BANK).T
    r = 1-q
    state = np.zeros(k)
    variance = r/(1-rho*rho)
    states, errors, svars = np.zeros((n,k)), np.zeros((n,k)), np.ones((n,k))
    for t in range(1,n):
        e = p[t]-p[t-1]-(rho-1)*state
        s = (rho-1)**2*variance+1.
        c = rho*(rho-1)*variance+r
        state = rho*state+c/s*e
        variance = np.maximum(rho*rho*variance+r-c*c/s,0.)
        states[t], errors[t], svars[t] = state,e,s
    return states, errors, svars

def reference(p: np.ndarray, window: int = 240):
    states, errors, svars = filter_bank(p)
    b = np.zeros((len(p),20)); selected = np.zeros(len(p),dtype=np.int64)
    z = np.zeros(len(p)); rho = np.zeros(len(p)); q = np.ones(len(p))
    choice = 0
    for t in range(len(p)):
        if t >= 60 and (t-60)%20 == 0:
            start=max(1,t-window)
            v=np.mean(errors[start:t]**2/svars[start:t],axis=0)
            score=np.log(svars[start:t]).sum(axis=0)+(t-start)*np.log(np.maximum(v,1e-12))
            choice=int(np.argmin(score))
        selected[t]=choice
        rho[t],q[t]=BANK[choice]
        z[t]=states[t,choice]
        b[t]=(rho[t]**np.arange(1,21)-1)*z[t]
    return dict(b=b,z=z,rho=rho,q=q,selected=selected)

def tensor_arrays(p: np.ndarray, ref: dict, end: int):
    """Complete mature origins, train/validation purge, training-only scale."""
    origins=np.arange(max(60,end-20-480),end-20)
    split=len(origins)-60
    if split-20 < 60:
        raise ValueError('Insufficient approved training support')
    tr=np.arange(split-20); va=np.arange(split,len(origins))
    x=np.array([np.diff(p[t-60:t+1]) for t in origins])
    y=np.array([p[t+1:t+21]-p[t] for t in origins])
    sd=max(float(np.std(x[tr])),1e-6)
    q=state_features(ref,origins,sd)
    b=ref['b'][origins]/sd
    tensors=[torch.tensor(a,dtype=torch.float32) for a in (x/sd,q,b,y/sd)]
    receipt=dict(fit_end=end,train_count=len(tr),validation_count=len(va),
                 first_training_origin=int(origins[tr[0]]),
                 last_training_origin=int(origins[tr[-1]]),
                 max_training_target=int(origins[tr[-1]]+20),
                 first_validation_origin=int(origins[va[0]]),
                 max_validation_target=int(origins[va[-1]]+20),scale=sd)
    if receipt['max_validation_target']>=end:
        raise ValueError('Future target')
    if receipt['max_training_target']>=receipt['first_validation_origin']:
        raise ValueError('Purging failed')
    return origins,tensors,tr,va,sd,receipt

def state_features(ref,origins,sd):
    return np.column_stack([np.tanh(ref['z'][origins]/sd),
                            ref['rho'][origins]*(1-ref['q'][origins]),
                            1-ref['q'][origins]])

class InteractionGRU(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode=mode
        self.encoder=nn.GRU(1,16,batch_first=True)
        self.U=nn.Linear(16,8,bias=False)
        self.V=nn.Linear(3,8,bias=False)
        self.bias=nn.Parameter(torch.zeros(8))
        self.O=nn.Linear(8,20,bias=False)
        nn.init.zeros_(self.O.weight)

    def hidden(self,x):
        h=self.encoder(x[:,:,None])[0][:,-1]
        z=self.encoder(torch.zeros((1,x.shape[1],1),dtype=x.dtype,device=x.device))[0][:,-1]
        return h-z

    def correction_from_hidden(self,h,q):
        a,c=self.U(h),self.V(q)
        f0=torch.tanh(self.bias)
        fx=torch.tanh(a+self.bias)-f0
        fq=torch.tanh(c+self.bias)-f0
        full=torch.tanh(a+c+self.bias)-f0
        feats={'RESIDUAL':fx,'ADDITIVE':fx+fq,'FULL':full,'MIXED':full-fx-fq}
        return self.O(feats[self.mode])

    def forward(self,x,q,b):
        correction=self.correction_from_hidden(self.hidden(x),q)
        return b+correction,correction

class PlainGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.GRU(1,16,batch_first=True)
        self.head=nn.Linear(16,20)
    def forward(self,x,q=None,b=None):
        pred=self.head(self.encoder(x[:,:,None])[0][:,-1])
        return pred,pred

def predict(model,sd,p,ref,origins):
    x=torch.tensor(np.array([np.diff(p[t-60:t+1])/sd for t in origins]),dtype=torch.float32)
    q=torch.tensor(state_features(ref,origins,sd),dtype=torch.float32)
    b=torch.tensor(ref['b'][origins]/sd,dtype=torch.float32)
    with torch.no_grad():
        pred,corr=model(x,q,b)
        if isinstance(model,InteractionGRU):
            return ref['b'][origins]+corr.numpy().astype(np.float64)*sd
        return pred.numpy().astype(np.float64)*sd
