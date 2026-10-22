"""Causal overlap-aware neural memory. No downloads or implicit file writes.

Only OVERLAP is a proposed mechanism. GLS, GRU, memory retrieval and the scalar
reference are existing tools; working overlap covariance is not a measured
market-error covariance. Negative GLS weights are allowed and reported.
"""
from __future__ import annotations
import csv,hashlib,io,os
from pathlib import Path
import numpy as np
import torch
from torch import nn

BASE_SHA='79c5003a2b38ca6cea540ada401eacaf1ae59439'
PROTOCOL_SHA='2920978de2401d33f659b91d05445df9efde6174'
PREFIX_SHA='1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'
SEEDS=(17,29,41,53)
MODES=('OVERLAP','DIAGONAL','SOFTMAX','SHUFFLED','THIN','FULL','GRU')
H=20; K=12
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

def overlap_matrix(origins,h):
    a=np.asarray(origins)
    return np.maximum(0,h-np.abs(a[...,None]-a[...,None,:]))/float(h)

def gls_weights(pi,omega,mask):
    """Exact equality-constrained SPD solve on valid candidates, differentiable.

    pi [N,K], omega [N,H,K,K], mask [N,K]. Invalid rows form identity blocks,
    RHS=0; they cannot act as duplicated retrieval evidence.
    """
    n,k=pi.shape;dtype=pi.dtype
    m=mask.to(dtype);inv=m/pi.clamp_min(1e-12)+(1-m)
    mat=omega*(m[:,None,:,None]*m[:,None,None,:])+torch.diag_embed(inv)[:,None,:,:]
    rhs=m[:,None,:,None].expand(n,omega.shape[1],k,1)
    sol=torch.linalg.solve(mat,rhs).squeeze(-1)
    total=sol.sum(-1,keepdim=True)
    return torch.where(total>0,sol/total.clamp_min(1e-300),torch.zeros_like(sol))

class MemoryData:
    """All raw features separated from supervised labels. Every key matures
    before its own query, even when the query belongs to the fitting subset.
    """
    def __init__(self,p,ref):
        self.p=np.asarray(p);self.ref=ref;n=len(p)
        self.x=np.zeros((n,60));self.local=np.ones(n)*1e-6
        for i in range(60,n):
            self.x[i]=np.diff(p[i-60:i+1]);self.local[i]=max(float(np.sqrt(np.mean(self.x[i]**2))),1e-6)
        nx=self.x/self.local[:,None]
        self.candidates={};self.masks={};self.max_target=-1
        for kind in ('regular','thin'):
            cand=np.zeros((n,K),dtype=np.int64);mask=np.zeros((n,K),dtype=bool)
            for i in range(60,n):
                pool=np.arange(max(60,i-480),i-H)
                if len(pool)==0:continue
                distances=((nx[pool]-nx[i])**2).mean(1)
                ranked=pool[np.argsort(distances,kind='stable')]
                chosen=[]
                for j in ranked:
                    if kind=='regular' or all(abs(int(j)-v)>=H for v in chosen):chosen.append(int(j))
                    if len(chosen)==K:break
                cand[i,:len(chosen)]=chosen;mask[i,:len(chosen)]=True
            self.candidates[kind]=cand;self.masks[kind]=mask
        self.values={}
        for kind in self.candidates:
            a=self.candidates[kind];mask=self.masks[kind];values=np.zeros((n,K,H))
            for i in range(60,n):
                for s,j in enumerate(a[i]):
                    if not mask[i,s]:continue
                    if j+H>=i:raise ValueError('Unmatured memory')
                    values[i,s]=(p[j+1:j+H+1]-p[j]-ref['b'][j])/self.local[j]
            self.values[kind]=values
    def tensors(self,sd,queries,mode):
        ids=np.asarray(queries,dtype=np.int64);last=int(ids.max())+1
        kind='thin' if mode=='THIN' else 'regular';a=self.candidates[kind][ids];mask=self.masks[kind][ids]
        # Encode the union once; padded origin zero is not observed evidence.
        allids=np.unique(np.r_[ids,a[mask]])
        idx=np.searchsorted(allids,ids);memidx=np.searchsorted(allids,a);memidx[~mask]=0
        q=np.column_stack([np.tanh(self.ref['mu'][allids]/sd),self.ref['rho'][allids]*(1-self.ref['q'][allids]),1-self.ref['q'][allids]])
        cov=np.stack([overlap_matrix(a,h) for h in range(1,H+1)],axis=1)
        if mode in ('DIAGONAL','THIN','SOFTMAX','FULL','GRU'):cov=np.broadcast_to(np.eye(K),cov.shape).copy()
        if mode=='SHUFFLED':
            for u,i in enumerate(ids):
                valid=int(mask[u].sum());per=np.r_[np.random.default_rng(1009+int(i)).permutation(valid),np.arange(valid,K)]
                cov[u]=cov[u][:,per][:,:,per]
        t=lambda x:torch.as_tensor(x,dtype=torch.float64)
        return dict(x=t(self.x[allids]/sd),q=t(q),query=torch.as_tensor(idx),key=torch.as_tensor(memidx),
                    mask=torch.as_tensor(mask),cov=t(cov),values=t(self.values[kind][ids]),
                    local=t(self.local[ids]/sd),base=t(self.ref['b'][ids]/sd),ids=ids)

class Forecast(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.encoder=nn.GRU(1,16,batch_first=True)
        if mode=='GRU':self.head=nn.Linear(16,H);return
        self.U=nn.Linear(16,8,bias=False);self.V=nn.Linear(3,8,bias=False)
        self.bias=nn.Parameter(torch.zeros(8));self.O=nn.Linear(8,H,bias=False);nn.init.zeros_(self.O.weight)
        # FULL also allocates the retrieval parameters so shared initialization is
        # exactly matched; their data gradients are zero there. Report this.
        self.metric=nn.Linear(19,8);nn.init.orthogonal_(self.metric.weight);nn.init.zeros_(self.metric.bias)
        self.beta=nn.Parameter(torch.zeros(H))
    def forward(self,batch,return_details=False):
        h=self.encoder(batch['x'][:,:,None])[0][:,-1];hq=h[batch['query']]
        if self.mode=='GRU':
            pred=self.head(hq)
            return (pred,dict(correction=pred)) if return_details else pred
        hzero=self.encoder(torch.zeros((1,60,1),dtype=h.dtype))[0][0,-1]
        qh=batch['q'][batch['query']]
        param=self.O(torch.tanh(self.U(hq-hzero)+self.V(qh)+self.bias)-torch.tanh(self.bias))
        if self.mode=='FULL':
            out=batch['base']+param
            return (out,dict(correction=param)) if return_details else out
        emb=torch.tanh(self.metric(torch.cat([h,batch['q']],dim=1)))
        dist=((emb[batch['query']][:,None]-emb[batch['key']])**2).mean(-1)/.25
        # all-invalid rows use a numerical dummy softmax that is multiplied out
        score=(-dist).masked_fill(~batch['mask'],-1e30)
        pi=torch.softmax(score,dim=-1)*batch['mask']
        if self.mode=='SOFTMAX':w=pi[:,None].expand(-1,H,-1)
        else:w=gls_weights(pi,batch['cov'],batch['mask'])
        mem=torch.einsum('nhk,nkh->nh',w,batch['values'])
        r=batch['local'][:,None]*(.5*torch.tanh(self.beta))*mem
        correction=param+r;out=batch['base']+correction
        if return_details:return out,dict(correction=correction,param=param,memory=r,w=w,pi=pi,beta=.5*torch.tanh(self.beta))
        return out

def make_model(mode,seed):
    configure();torch.manual_seed(seed);return Forecast(mode)
