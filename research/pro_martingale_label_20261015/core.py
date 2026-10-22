"""Mean-preserving distributional functional predictor and fixed alternatives.
The probability outputs are smoothed regression labels, not calibrated price laws.
"""
from __future__ import annotations
from common import *
from scipy.linalg import expm
from scipy.special import ndtr

CATEGORICAL=('MP_DIFF','TWOHOT','GAUSS','CAT_MSE','IND_DIFF')
J=17

def initial_basis(rank=6):
    knots=np.rint(np.geomspace(1,H,rank)).astype(int)
    return torch.as_tensor(np.stack([np.interp(np.arange(1,H+1),np.r_[0,knots],np.r_[0,row]) for row in np.eye(rank)],1))

def two_hot(values,atoms):
    """All training values must be inside support. No silent clipping."""
    n,h=values.shape;j=atoms.shape[1];out=np.zeros((n,h,j))
    for k in range(h):
        y=values[:,k];c=atoms[k]
        if np.any(y<c[0]-1e-12) or np.any(y>c[-1]+1e-12):raise ValueError('Unsupported training value')
        hi=np.searchsorted(c,y,side='right').clip(1,j-1);lo=hi-1
        frac=(y-c[lo])/(c[hi]-c[lo]);out[np.arange(n),k,lo]=1-frac;out[np.arange(n),k,hi]=frac
    return out

def generator(c):
    n=len(c);Q=np.zeros((n,n))
    for j in range(1,n-1):
        dl=c[j]-c[j-1];dr=c[j+1]-c[j]
        Q[j,j-1]=1/(dl*(dl+dr));Q[j,j+1]=1/(dr*(dl+dr));Q[j,j]=-Q[j].sum()
    return Q

def gaussian_probs(values,atoms,sigma):
    edges=(atoms[:,:-1]+atoms[:,1:])/2
    edges=np.column_stack([np.full(H,-np.inf),edges,np.full(H,np.inf)])
    q=np.diff(ndtr((edges[None]-values[:,:,None])/sigma[None,:,None]),axis=-1)
    return q

def label_system(target):
    """Only the current training labels set support and smoothing."""
    t=target/2
    amp=np.maximum(2.,1.05*np.max(np.abs(t),axis=0))
    u=np.linspace(-1,1,J);atoms=amp[:,None]*(np.sign(u)*u*u)
    width=np.maximum(np.sqrt(np.mean(t*t,0)),.1);tau=(.3*width)**2
    kernels=np.stack([expm(tau[h]*generator(atoms[h])) for h in range(H)])
    q0=two_hot(t,atoms);mp=np.einsum('nhj,hjk->nhk',q0,kernels)
    gauss=gaussian_probs(t,atoms,np.sqrt(tau))
    prior=gaussian_probs(np.zeros((1,H)),atoms,width)[0]
    prior=np.maximum(prior,1e-14);prior/=prior.sum(-1,keepdims=True)
    if not np.isfinite(kernels).all() or mp.min()<-1e-10:raise FloatingPointError('Invalid diffusion kernel')
    error=np.max(np.abs((mp*atoms).sum(-1)-t))
    if error>1e-9:raise ArithmeticError(f'Mean preservation failure {error}')
    diagnostic=dict(target_mean_max_error=float(error),mass_max_error=float(np.max(np.abs(mp.sum(-1)-1))),
        gaussian_bias_rms=float(np.sqrt(np.mean(((gauss*atoms).sum(-1)-t)**2))),
        gaussian_bias_max=float(np.max(np.abs((gauss*atoms).sum(-1)-t))),
        minimum_probability=float(mp.min()),mean_target_entropy=float(-np.mean(np.sum(mp*np.log(np.maximum(mp,1e-300)),axis=-1))))
    return atoms,np.log(prior),dict(MP_DIFF=mp,IND_DIFF=mp,TWOHOT=q0,GAUSS=gauss,CAT_MSE=mp),kernels,diagnostic

class Forecast(nn.Module):
    def __init__(self,mode,atoms=None,prior=None):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;hidden=22 if mode=='WIDE_RANK6' else 16
        self.encoder=nn.GRU(1,hidden,batch_first=True)
        if mode=='GRU':self.head=nn.Linear(hidden,H);return
        self.U=nn.Linear(hidden,8,bias=False);self.V=nn.Linear(3,8,bias=False);self.bias=nn.Parameter(torch.zeros(8))
        if mode in CATEGORICAL:
            if atoms is None:atoms=np.zeros((H,J))
            if prior is None:prior=np.zeros((H,J))
            self.register_buffer('atoms',torch.as_tensor(atoms));self.register_buffer('prior',torch.as_tensor(prior))
            rank=H if mode=='IND_DIFF' else 6
            self.logit_head=nn.Linear(8,rank*J);nn.init.zeros_(self.logit_head.weight);nn.init.zeros_(self.logit_head.bias)
            self.register_buffer('basis',torch.eye(H) if mode=='IND_DIFF' else initial_basis(6))
        else:
            self.coefficients=nn.Linear(8,6);nn.init.zeros_(self.coefficients.weight);nn.init.zeros_(self.coefficients.bias)
            if mode=='FIXED6':self.register_buffer('basis',initial_basis(6))
            else:self.basis=nn.Parameter(initial_basis(6))
            self.drift=nn.Linear(8,1);nn.init.zeros_(self.drift.weight);nn.init.zeros_(self.drift.bias)
    def features(self,x,q):
        h=self.encoder(x[:,:,None])[0][:,-1]
        hz=self.encoder(torch.zeros((1,60,1),dtype=x.dtype,device=x.device))[0][0,-1]
        return torch.tanh(self.U(h-hz)+self.V(q)+self.bias)-torch.tanh(self.bias)
    def forward(self,x,q,base,details=False):
        if self.mode=='GRU':
            c=self.head(self.encoder(x[:,:,None])[0][:,-1]);info=dict(correction=c)
            return (c,info) if details else c
        f=self.features(x,q)
        if self.mode in CATEGORICAL:
            coef=self.logit_head(f).reshape(-1,self.basis.shape[1],J)
            logits=torch.einsum('hr,nrj->nhj',self.basis,coef)+self.prior
            logprob=torch.log_softmax(logits,-1);prob=logprob.exp();c=(prob*self.atoms).sum(-1)
            info=dict(correction=c,logprob=logprob,prob=prob)
        else:
            coeff=3*torch.tanh(self.coefficients(f));d=torch.tanh(self.drift(f)).squeeze(-1)
            c=coeff@self.basis.T+d[:,None]*torch.arange(1,H+1,dtype=x.dtype)/H
            info=dict(correction=c,coefficients=coeff,drift=d)
        return (base+c,info) if details else base+c

def make_model(mode,seed,atoms=None,prior=None):
    configure();torch.manual_seed(int(seed));return Forecast(mode,atoms,prior)
