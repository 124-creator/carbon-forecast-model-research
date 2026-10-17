"""Causal prefix infrastructure reused as inspected; duration-dependent response decoder.
"""
from __future__ import annotations
import csv,hashlib,io,os
from pathlib import Path
import numpy as np
import torch
from torch import nn

BASE_SHA='ba282791e1900aeed8c120535798e8532a4f5670'
PROTOCOL_SHA='618250b93f992c574350f2a08d0500b7e964c834'
PREFIX_SHA='1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'
SEEDS=(17,29,41,53)
MODES=('RENEWAL','CLOCK','MARKOV','RANK4','RANK6','FIXED6','DIRECT','GRU')
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


def stochastic_matrix(logits):
    return (1-EPS)*torch.softmax(logits,-1)+EPS/K

def markov_curve(pi,P,v,drift,steps=H):
    at=pi;initial=(pi*v).sum(-1);ys=[];occupancies=[]
    for h in range(1,steps+1):
        at=torch.bmm(at[:,None,:],P).squeeze(1)
        ys.append((at*v).sum(-1)-initial+h/H*drift);occupancies.append(at)
    return torch.stack(ys,-1),torch.stack(occupancies,1)

def renewal_curve(pi,P,v,drift,beta,clock=False,steps=H):
    """Exact finite-horizon sum over state-age paths, not inferred economic ages."""
    batch,states=pi.shape
    exits=1-P.diagonal(dim1=-2,dim2=-1)
    eye=torch.eye(states,dtype=P.dtype,device=P.device)
    destination=P*(1-eye)/exits[:,:,None]
    logodds=torch.log(exits)-torch.log1p(-exits)
    # No age cap is reached before the final requested step.
    ages=torch.arange(steps+1,dtype=P.dtype,device=P.device)
    age_basis=torch.log1p(ages)/np.log(H)
    mass=torch.cat((pi[:,:,None],pi.new_zeros(batch,states,steps)),dim=2)
    first=(pi*v).sum(-1);ys=[];occupancies=[];mean_ages=[]
    duration_hazard=torch.sigmoid(logodds[:,:,None]+beta[:,:,None]*age_basis)
    for h in range(steps):
        hazard=(torch.sigmoid(logodds+beta*(np.log1p(h)/np.log(H)))[:,:,None]
                if clock else duration_hazard)
        leaving=(mass*hazard).sum(-1)
        arrivals=torch.bmm(leaving[:,None,:],destination).squeeze(1)
        staying=mass*(1-hazard)
        mass=torch.cat((arrivals[:,:,None],staying[:,:,:-1]),dim=2)
        at=mass.sum(-1)
        occupancies.append(at);mean_ages.append((mass*ages).sum((-1,-2)))
        ys.append((at*v).sum(-1)-first+(h+1)/H*drift)
    return torch.stack(ys,-1),dict(occupancy=torch.stack(occupancies,1),
        final_age_mass=mass,mean_age=torch.stack(mean_ages,1),
        beta=beta,exit_probability=exits,destination=destination)

def initial_basis(rank):
    """Piecewise-linear forecast-basis control, not an N-BEATS reproduction."""
    knots=np.rint(np.geomspace(1,H,rank)).astype(int)
    if len(np.unique(knots))!=rank:raise ValueError('duplicate predetermined knots')
    points=np.r_[0,knots]
    out=np.stack([np.interp(np.arange(1,H+1),points,np.r_[0,np.eye(rank)[i]])
                  for i in range(rank)],axis=1)
    return torch.as_tensor(out,dtype=torch.get_default_dtype())

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
        if mode in ('RANK4','RANK6','FIXED6'):
            rank=4 if mode=='RANK4' else 6
            self.coefficients=nn.Linear(8,rank)
            nn.init.zeros_(self.coefficients.weight);nn.init.zeros_(self.coefficients.bias)
            B=initial_basis(rank)
            if mode=='FIXED6':self.register_buffer('basis',B)
            else:self.basis=nn.Parameter(B)
            self.drift=nn.Linear(8,1);nn.init.zeros_(self.drift.weight);nn.init.zeros_(self.drift.bias)
            return
        # Keep the original Stage L construction/random draws exactly for common parts.
        self.mix=nn.Linear(8,3);nn.init.zeros_(self.mix.weight);nn.init.zeros_(self.mix.bias)
        self.initial=nn.Linear(8,K)
        self.payoff=nn.Linear(8,K);self.drift=nn.Linear(8,1)
        nn.init.zeros_(self.payoff.weight);nn.init.zeros_(self.payoff.bias)
        nn.init.zeros_(self.drift.weight);nn.init.zeros_(self.drift.bias)
        cycle=torch.roll(torch.eye(K),shifts=1,dims=1)
        bases=torch.stack([2*torch.eye(K),.5*torch.eye(K)+1.5*cycle,
                           .5*torch.eye(K)+1.5*cycle.T])
        self.logits=nn.Parameter(bases+.05*torch.randn_like(bases))
        if mode in ('RENEWAL','CLOCK'):
            self.duration=nn.Linear(8,K)
            nn.init.zeros_(self.duration.weight);nn.init.zeros_(self.duration.bias)
    def features(self,x,q):
        h=self.encoder(x[:,:,None])[0][:,-1]
        hz=self.encoder(torch.zeros((1,60,1),dtype=x.dtype,device=x.device))[0][0,-1]
        return torch.tanh(self.U(h-hz)+self.V(q)+self.bias)-torch.tanh(self.bias)
    def forward(self,x,q,base,details=False):
        if self.mode=='GRU':
            pred=self.head(self.encoder(x[:,:,None])[0][:,-1])
            return (pred,dict(correction=pred)) if details else pred
        f=self.features(x,q)
        if self.mode=='DIRECT':
            correction=self.head(f);info={}
        elif self.mode in ('RANK4','RANK6','FIXED6'):
            coeff=3*torch.tanh(self.coefficients(f));d=torch.tanh(self.drift(f)).squeeze(-1)
            h=torch.arange(1,H+1,dtype=x.dtype,device=x.device)/H
            correction=coeff@self.basis.T+d[:,None]*h
            info=dict(coefficients=coeff,drift=d)
        else:
            weights=torch.softmax(self.mix(f),-1)
            P=stochastic_matrix(torch.einsum('nm,mij->nij',weights,self.logits))
            pi=torch.softmax(self.initial(f),-1)
            v=3*torch.tanh(self.payoff(f));v=v-v.mean(-1,keepdim=True)
            d=torch.tanh(self.drift(f)).squeeze(-1)
            if self.mode=='MARKOV':
                correction,occupancy=markov_curve(pi,P,v,d)
                info=dict(occupancy=occupancy)
            else:
                beta=2*torch.tanh(self.duration(f))
                correction,info=renewal_curve(pi,P,v,d,beta,clock=self.mode=='CLOCK')
            info.update(P=P,pi=pi,v=v,drift=d,mix=weights)
        info['correction']=correction
        return (base+correction,info) if details else base+correction

def make_model(mode,seed):
    configure();torch.manual_seed(int(seed));return Forecast(mode)
