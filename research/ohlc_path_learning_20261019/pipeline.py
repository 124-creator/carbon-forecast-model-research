"""Chronological data preparation. Every feature at origin o uses rows <= o.

All horizons count observed records; no dates are imputed for Quantum. This
module never downloads data and refuses Ding observations after 2023-12-31.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch

HORIZONS=(1,5,20)
LOOKBACK=20
FIRST_ORIGIN=60

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_source(path, source):
    d=pd.read_csv(path,keep_default_na=True)
    assert list(d.source_index)==list(range(len(d))), 'Source index must be unmodified contiguous order'
    for c in ('price','high','low'):
        assert np.isfinite(d[c]).all() and (d[c]>0).all(), c
    assert (d.high>=d.low).all(), 'High/low ordering, not P containment, is required'
    if source=='ding':
        dates=pd.to_datetime(d.date)
        assert dates.is_monotonic_increasing and dates.is_unique
        assert (dates<=pd.Timestamp('2023-12-31')).all()
    if source=='quantum':
        assert d.date.isna().all(), 'Quantum dates must not be invented'
    return d

def folds_for(d,source):
    if source=='quantum':
        assert len(d)==1805
        return [('Q1',1000,1250),('Q2',1250,1500),('Q3',1500,len(d)-20)]
    if source=='ding':
        assert len(d)==772
        dates=pd.to_datetime(d.date)
        starts=[int(np.searchsorted(dates.values,np.datetime64(t))) for t in ('2022-07-01','2023-01-01','2023-07-01')]
        return [(f'D{i+1}',a,b) for i,(a,b) in enumerate(zip(starts,starts[1:]+[len(d)-20]))]
    raise ValueError(source)

class OriginData:
    def __init__(self,d):
        self.frame=d
        p=d.price.to_numpy(float); high=d.high.to_numpy(float); low=d.low.to_numpy(float)
        n=len(p); r=np.r_[0.,np.diff(np.log(p))]
        basis=np.log(np.sqrt(high*low)/p)
        radius=.5*np.log(high/low)
        op=d.open.to_numpy(float); vol=d.volume.to_numpy(float)
        omask=np.isfinite(op)&(op>0); vmask=np.isfinite(vol)&(vol>=0)
        ogap=np.zeros(n); ogap[omask]=np.log(op[omask]/p[omask])
        lv=np.log1p(np.where(vmask,vol,0.))
        vchange=np.r_[0.,np.diff(lv)]
        vchange[~(vmask&np.r_[False,vmask[:-1]])]=0.
        self.origin=np.arange(FIRST_ORIGIN,n-20,dtype=int)
        self.x=[]; self.context=[]; self.p0=[]; self.scale=[]; self.basis0=[]; self.radius0=[]; self.y=[]
        for o in self.origin:
            # Same causal price scale is supplied to every model, including price-only controls.
            scale=float(np.clip(np.std(r[o-59:o+1],ddof=1),.003,.2))
            ix=np.arange(o-LOOKBACK+1,o+1)
            features=np.stack((r[ix]/scale,basis[ix]/scale,radius[ix]/scale,ogap[ix]/scale,
                vchange[ix],omask[ix].astype(float),np.ones(LOOKBACK),vmask[ix].astype(float)),axis=1)
            self.x.append(np.clip(features,-10.,10.))
            self.context.append([np.log(scale/.02),np.sum(r[o-19:o+1])/(scale*np.sqrt(20))])
            self.p0.append(p[o]); self.scale.append(scale)
            self.basis0.append(basis[o]/scale); self.radius0.append(radius[o]/scale)
            self.y.append([[p[o+h],high[o+1:o+h+1].max(),low[o+1:o+h+1].min()] for h in HORIZONS])
        for name in ('x','context','p0','scale','basis0','radius0','y'):
            setattr(self,name,np.asarray(getattr(self,name),dtype=np.float64))
        self.valid=np.isfinite(self.y)
        assert self.valid.all(), 'Primary sources must have complete P/H/L'
        self.dates=np.asarray([str(d.date.iloc[o]) if pd.notna(d.date.iloc[o]) else '' for o in self.origin])

    def positions(self,start,end):
        ix=np.flatnonzero((self.origin>=start)&(self.origin<end))
        assert len(ix)==end-start, (start,end,len(ix))
        return ix

@dataclass
class Prepared:
    data: OriginData
    train: np.ndarray
    xmean: np.ndarray
    xstd: np.ndarray
    cmean: np.ndarray
    cstd: np.ndarray
    denominator: np.ndarray

    @classmethod
    def fit(cls,data,train):
        x=data.x[train]; c=data.context[train]
        # Summary losses use fixed, training-only RMS displacement scales.
        den=np.sqrt(np.mean((data.y[train]-data.p0[train,None,None])**2,axis=0))
        den=np.maximum(den,np.mean(data.p0[train])*.001)
        return cls(data,train,x.mean((0,1)),np.maximum(x.std((0,1)),.1),c.mean(0),np.maximum(c.std(0),.1),den)

    def arrays(self,ix):
        d=self.data
        return {'x':np.clip((d.x[ix]-self.xmean)/self.xstd,-10.,10.),
                'context':np.clip((d.context[ix]-self.cmean)/self.cstd,-10.,10.),
                'p0':d.p0[ix],'scale':d.scale[ix],'basis0':d.basis0[ix],'radius0':d.radius0[ix],
                'y':d.y[ix],'valid':d.valid[ix]}

    def tensors(self,ix):
        return {k:torch.as_tensor(v,dtype=torch.bool if k=='valid' else torch.float64) for k,v in self.arrays(ix).items()}

    def classical(self,ix,price_only=False):
        a=self.arrays(ix)
        x=a['x'][:,:,:1] if price_only else a['x']
        return np.concatenate((x.reshape(len(ix),-1),a['context']),axis=1)

    def state(self):
        return {k:np.asarray(getattr(self,k)) for k in ('xmean','xstd','cmean','cstd','denominator')}
