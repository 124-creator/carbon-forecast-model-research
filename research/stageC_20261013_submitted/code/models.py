"""Stage C: causal, matched-budget point forecasts. No old prediction inputs."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
from torch import nn
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
H=np.array([1,5,20],dtype=int)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

@dataclass
class Data:
    p:np.ndarray
    dates:np.ndarray
    first:int
    scales:np.ndarray
    x:np.ndarray
    q:np.ndarray
    xr:np.ndarray
    xd:np.ndarray
    y:np.ndarray

def make_data(p,dates,first):
    p=np.asarray(p,dtype=float);dates=np.asarray(dates,dtype='datetime64[D]')
    if len(p)!=len(dates) or first<81 or not np.isfinite(p).all():raise ValueError('invalid data')
    if not np.all(np.diff(dates).astype(int)>0):raise ValueError('dates not unique/ascending')
    scales=np.array([np.mean((p[h:first]-p[:first-h])**2) for h in H])
    if np.min(scales)<=0:raise ValueError('zero loss scale')
    n=len(p);x=np.full((n,61),np.nan);xr=x.copy();xd=np.full((n,60),np.nan);q=np.full(n,np.nan)
    y=np.full((n,3),np.nan)
    for t in range(60,n):
        q[t]=max(np.sqrt(np.mean(np.diff(p[t-60:t+1])**2)),0.05*np.sqrt(scales[0]))
        gap=np.log1p(int((dates[t]-dates[t-1]).astype(int)))
        xr[t]=np.r_[p[t-59:t+1]-p[t],gap]
        x[t]=np.r_[xr[t,:60]/q[t],gap]
        xd[t]=np.diff(p[t-60:t+1])[::-1]
        for j,h in enumerate(H):
            if t+h<n:y[t,j]=p[t+h]-p[t]
    return Data(p,dates,first,scales,x,q,xr,xd,y)

def training_indices(t,window=512):
    end=t-20
    idx=np.arange(max(60,end-window+1),end+1)
    if len(idx)<1:raise ValueError('insufficient matured labels')
    return idx

class ResponseNet(nn.Module):
    def __init__(self,kind='conditional'):
        super().__init__();self.kind=kind
        self.trunk=nn.Sequential(nn.Linear(61,32),nn.GELU(),nn.Linear(32,16),nn.GELU())
        self.head=nn.Linear(16,3 if kind in ['free','conditional','no_gap','no_drift'] else 2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        z=float(np.log((10-.5)/(40-10)))
        if kind in ['conditional','no_gap','no_drift']:
            with torch.no_grad():self.head.bias[2]=z
        elif kind=='global':self.logit_tau=nn.Parameter(torch.tensor(z,dtype=torch.float32))
        self.register_buffer('h',torch.tensor(H,dtype=torch.float32))
    def forward(self,x,q,return_tau=False):
        if self.kind=='no_gap':x=torch.cat([x[:,:60],torch.zeros_like(x[:,60:61])],dim=1)
        a=self.head(self.trunk(x))
        if self.kind=='free':out=a;tau=torch.full_like(a[:,0],float('nan'))
        else:
            if self.kind in ['conditional','no_gap','no_drift']:tau=.5+39.5*torch.sigmoid(a[:,2])
            elif self.kind=='global':tau=(.5+39.5*torch.sigmoid(self.logit_tau)).expand(len(x))
            elif self.kind=='fixed':tau=torch.full_like(a[:,0],10.)
            else:raise ValueError(self.kind)
            trend=a[:,0:1]*(self.h[None,:]/20)
            if self.kind=='no_drift':trend=trend*0
            out=trend+a[:,1:2]*(1-torch.exp(-self.h[None,:]/tau[:,None]))
        out=out*q[:,None]
        return (out,tau) if return_tau else out

def neural_forecasts(d:Data,kind,lr,wd,seed,initial_steps=200,refit_steps=20):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=ResponseNet(kind);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    x=torch.tensor(np.nan_to_num(d.x).astype('float32'));q=torch.tensor(np.nan_to_num(d.q).astype('float32'))
    y=torch.tensor(np.nan_to_num(d.y).astype('float32'));s=torch.tensor(d.scales.astype('float32'))
    pred=np.full((len(d.p),3),np.nan);tau=np.full(len(d.p),np.nan)
    fitlog=[]
    for t in range(d.first,len(d.p)):
        if (t-d.first)%20==0:
            idx=training_indices(t);steps=initial_steps if t==d.first else refit_steps
            if not np.isfinite(d.y[idx]).all():raise RuntimeError('immature labels passed to fit')
            model.train()
            for _ in range(steps):
                b=rng.choice(idx,size=min(128,len(idx)),replace=False)
                opt.zero_grad(set_to_none=True);out=model(x[b],q[b]);loss=((out-y[b])**2/s).mean()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            fitlog.append((t,int(idx.max()+20),int(idx.min()),int(idx.max()),steps))
        model.eval()
        with torch.no_grad():
            pp,tt=model(x[t:t+1],q[t:t+1],True);pred[t]=pp.numpy()[0];tau[t]=float(tt[0])
    return pred,tau,{'fit_events':fitlog,'parameters':sum(p.numel() for p in model.parameters()),'updates':sum(v[-1] for v in fitlog),'seed':seed}

def baseline_forecasts(d,family,params):
    n=len(d.p);out=np.full((n,3),np.nan)
    if family=='RW':out[d.first:]=0;return out
    if family=='drift':
        w=params['window']
        for t in range(d.first,n):out[t]=H*(d.p[t]-d.p[max(0,t-w)])/min(w,t)
        return out
    if family in ['SES','Holt']:
        a=params['alpha'];b=params.get('beta',0);phi=params.get('phi',0)
        ell=d.p[0];trend=0.
        for t in range(1,n):
            new=a*d.p[t]+(1-a)*(ell+phi*trend)
            trend=b*(new-ell)+(1-b)*phi*trend;ell=new
            if t>=d.first:out[t]=ell+trend*np.array([sum(phi**k for k in range(1,h+1)) for h in H])-d.p[t]
        return out
    roots=np.sqrt(d.scales)
    x=d.xd[:,:params['lag']] if family=='Ridge_diff' else (np.c_[d.xr[:,60-params['lag']:60],d.xr[:,60]] if family=='Ridge_level' else d.xr)
    for t in range(d.first,n):
        if (t-d.first)%20==0:
            idx=training_indices(t);scaler=StandardScaler().fit(x[idx]);X=scaler.transform(x[idx]);Y=d.y[idx]/roots
            if family.startswith('Ridge'):
                models=Ridge(alpha=params['alpha']*len(idx)).fit(X,Y)
            else:
                models=[]
                for j in range(3):
                    if family=='HGBT':m=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=params['leaves'],learning_rate=params['lr'],l2_regularization=params['l2'],min_samples_leaf=20,early_stopping=False,random_state=11)
                    elif family=='SVR':m=SVR(C=params['C'],gamma=params['gamma'],epsilon=params['epsilon'])
                    else:raise ValueError(family)
                    m.fit(X,Y[:,j]);models.append(m)
        Xtest=scaler.transform(x[t:t+1])
        out[t]=(models.predict(Xtest)[0] if family.startswith('Ridge') else np.array([m.predict(Xtest)[0] for m in models]))*roots
    return out

def origin_mask(d,year):
    yr=pd.DatetimeIndex(d.dates).year.to_numpy()
    return np.array([i for i in range(d.first,len(d.p)-20) if yr[i]==year and yr[i+20]==year],dtype=int)

def score(d,pred,years):
    # pred [seeds,n,3] or [n,3]; average losses, never ensemble predictions.
    if pred.ndim==2:pred=pred[None]
    rows=[]
    for year in years:
        ids=origin_mask(d,year)
        if not len(ids):continue
        pp=pred[:,ids,:];yy=d.y[ids]
        if not np.isfinite(pp).all():raise ValueError('missing scoring forecasts')
        mse=np.mean((pp-yy[None])**2,axis=(0,1));mae=np.mean(np.abs(pp-yy[None]),axis=(0,1));rw=np.mean(yy**2,axis=0)
        for j,h in enumerate(H):rows.append({'year':int(year),'horizon':int(h),'n':len(ids),'mse':mse[j],'mae':mae[j],'rmse':float(np.sqrt(mse[j])),'rw_mse':rw[j],'relative_mse':mse[j]/rw[j],'nmse':mse[j]/d.scales[j]})
    return rows

def tree_forecasts_batched(d,params):
    """Identical HGBT fits; batch independent predictions between fixed refits."""
    n=len(d.p);out=np.full((n,3),np.nan);roots=np.sqrt(d.scales);x=d.xr
    for t in range(d.first,n,20):
        idx=training_indices(t);ss=StandardScaler().fit(x[idx]);X=ss.transform(x[idx]);Y=d.y[idx]/roots
        end=min(t+20,n);xx=ss.transform(x[t:end])
        for j in range(3):
            m=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=params['leaves'],learning_rate=params['lr'],l2_regularization=params['l2'],min_samples_leaf=20,early_stopping=False,random_state=11)
            m.fit(X,Y[:,j]);out[t:end,j]=m.predict(xx)*roots[j]
    return out
