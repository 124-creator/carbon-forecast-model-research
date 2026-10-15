"""Causal prefix infrastructure reused as inspected; duration-dependent response decoder.
"""
from __future__ import annotations
import csv,hashlib,io,os
from pathlib import Path
import numpy as np
import torch
from torch import nn

BASE_SHA='ba282791e1900aeed8c120535798e8532a4f5670'
PROTOCOL_SHA='a0a061da637d5f3a72a44b02d29dc4cba869eb01'
PREFIX_SHA='1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'
SEEDS=(17,29,41,53)
MODES=('MP_DIFF','TWOHOT','GAUSS','CAT_MSE','IND_DIFF','FIXED6','RANK6','WIDE_RANK6','GRU')
H=20; K=4; EPS=.02
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

