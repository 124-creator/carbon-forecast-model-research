"""Joint *reported* price/range forecasting, not an identified transaction path.

All stochastic calculations use float64. The reported price is deliberately not
constrained to lie in the reported daily interval. Max/min refer to daily
reported high/low series, not a reconstructed tick path.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

HORIZONS = (1, 5, 20)
H = 20


def inverse_softplus(x):
    # Exact zero radius has inverse -infinity; use a stated finite initial proxy.
    x = torch.clamp(x, min=1e-5)
    return x + torch.log(-torch.expm1(-x))


def sobol_normal(n: int, seed: int):
    if n & (n-1):
        raise ValueError('Sobol sample size must be a power of two')
    u = torch.quasirandom.SobolEngine(H*4, scramble=True, seed=seed).draw_base2(int(math.log2(n))).double()
    return (math.sqrt(2)*torch.erfinv(2*u.clamp(1e-10, 1-1e-10)-1)).reshape(n,H,4)


def summaries(price, high, low):
    """Inputs N x S x day; output N x S x horizon x (P,U,L)."""
    ix = torch.tensor([h-1 for h in HORIZONS], device=price.device)
    return torch.stack((price.index_select(2,ix), high.cummax(2).values.index_select(2,ix),
                        low.cummin(2).values.index_select(2,ix)), dim=-1)


class Encoder(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.gru = nn.GRU(8, width, batch_first=True)
        self.mix = nn.Sequential(nn.Linear(width+2,width),nn.Tanh())

    def forward(self, x, context):
        _, h = self.gru(x)
        return self.mix(torch.cat((h[-1],context),dim=-1))


class PointGRU(nn.Module):
    def __init__(self, price_only=False):
        super().__init__()
        self.price_only = price_only
        self.encoder = Encoder()
        self.head = nn.Linear(16,3)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, context, p0, scale, basis0, radius0, eps=None):
        if self.price_only:
            x = torch.cat((x[:,:,:1],torch.zeros_like(x[:,:,1:])),dim=-1)
        raw = self.head(self.encoder(x,context))
        # Bounded log adjustment protects only the numerical price domain.
        raw_adjustment = scale[:,None]*raw*torch.tensor(HORIZONS,device=x.device).sqrt()
        adjustment = torch.clamp(raw_adjustment,-3.,3.)
        return {'mean':p0[:,None]*torch.exp(adjustment), 'samples':None,
                'clip_count':(raw_adjustment.abs()>3.).sum()}


class SummaryState(nn.Module):
    """A low-dimensional conditional Markov response shared by P/d/q.

    d = log(sqrt(high*low)/P), q = .5 log(high/low).
    These are observed statistical coordinates, without an efficient-price or
    measurement-error interpretation. A future state's mean controls the
    signed mean return; its innovations also affect the range coordinates.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(16,13)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, context, p0, scale, basis0, radius0, eps):
        v = self.head(self.encoder(x,context))
        drift = .5*torch.tanh(v[:,0])
        z0 = 2.*torch.tanh(v[:,1])
        rho = .98*torch.sigmoid(v[:,2]+1.)
        loading = .5*torch.tanh(v[:,3]+.2)
        sigma = .05+1.95*torch.sigmoid(v[:,4])
        dbar = 2.*torch.tanh(v[:,5])
        rhod = .98*torch.sigmoid(v[:,6]+1.)
        loadd = torch.tanh(v[:,7])
        sigmad = .01+.49*torch.sigmoid(v[:,8]-1.)
        qbar = v[:,9]-.7
        rhoq = .98*torch.sigmoid(v[:,10]+1.)
        loadq = torch.tanh(v[:,11])
        sigmaq = .01+.49*torch.sigmoid(v[:,12]-1.)

        days = torch.arange(1,H+1,device=x.device,dtype=x.dtype)
        row = torch.arange(H,device=x.device)[:,None]
        col = torch.arange(H,device=x.device)[None,:]
        power = (row-col).clamp(min=0)
        K = torch.sqrt(1-rho[:,None,None]**2)*rho[:,None,None]**power[None]
        K = K*(row>=col)[None]
        Kcum = K.cumsum(1)
        zmean = z0[:,None]*rho[:,None]**days
        znoise = torch.einsum('njk,sk->nsj',K,eps[:,:,0])
        state = zmean[:,None,:]+znoise
        meanlog = scale[:,None]*(drift[:,None]*days+loading[:,None]*zmean.cumsum(1))
        variance = scale[:,None]**2*(loading[:,None]**2*Kcum.square().sum(2)+sigma[:,None]**2*days)
        noise = scale[:,None,None]*(loading[:,None,None]*znoise.cumsum(2)+sigma[:,None,None]*eps[:,:,1].cumsum(1)[None])
        price = p0[:,None,None]*torch.exp((meanlog[:,None,:]+noise-.5*variance[:,None,:]).clamp(-8.,8.))
        # Numerical clipping is monitored by the runner and should be inactive.
        db = dbar[:,None]+rhod[:,None]**days*(basis0[:,None]-dbar[:,None])
        d = scale[:,None,None]*(db[:,None,:]+loadd[:,None,None]*state+sigmad[:,None,None]*eps[:,:,2][None])
        qb = qbar[:,None]+rhoq[:,None]**days*(inverse_softplus(radius0)[:,None]-qbar[:,None])
        q = scale[:,None,None]*F.softplus(qb[:,None,:]+loadq[:,None,None]*state+sigmaq[:,None,None]*eps[:,:,3][None])
        high = price*torch.exp((d+q).clamp(-8.,8.))
        low = price*torch.exp((d-q).clamp(-8.,8.))
        ix = torch.tensor([h-1 for h in HORIZONS],device=x.device)
        mean = p0[:,None]*torch.exp(meanlog.index_select(1,ix))
        clip_count = ((meanlog[:,None,:]+noise-.5*variance[:,None,:]).abs()>8.).sum()+((d+q).abs()>8.).sum()+((d-q).abs()>8.).sum()
        return {'mean':mean,'samples':summaries(price,high,low),'clip_count':clip_count,
                'daily':(price,high,low),'meanlog':meanlog,'variance':variance}


class FreePath(nn.Module):
    """Flexible joint daily decoder; same summaries and score as the candidate.

    One common Gaussian factor and independent daily innovations permit joint
    dependence. Daily coefficients are free instead of following the scalar
    Markov response. It has more parameters; this is a strong flexible control,
    not an assertion of identical parameter counts or universal expressivity.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(16,H*9)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self,x,context,p0,scale,basis0,radius0,eps):
        v = self.head(self.encoder(x,context)).reshape(-1,H,9)
        mu = .5*torch.tanh(v[:,:,0])
        loading = .5*torch.tanh(v[:,:,1]+.2)
        sigma = .05+1.95*torch.sigmoid(v[:,:,2])
        common = eps[:,0,0]
        meanlog = scale[:,None]*mu.cumsum(1)
        variance = scale[:,None]**2*(loading.cumsum(1).square()+sigma.square().cumsum(1))
        noise = scale[:,None,None]*(loading.cumsum(1)[:,None,:]*common[None,:,None]+
                                   (sigma[:,None,:]*eps[:,:,1][None]).cumsum(2))
        lp = meanlog[:,None,:]+noise-.5*variance[:,None,:]
        price = p0[:,None,None]*torch.exp(lp.clamp(-8.,8.))
        decay = torch.exp(-torch.arange(1,H+1,device=x.device,dtype=x.dtype)/5.)
        d = scale[:,None,None]*(basis0[:,None,None]*decay[None,None,:]+2*torch.tanh(v[:,:,3])[:,None,:]+
             torch.tanh(v[:,:,4])[:,None,:]*common[None,:,None]+(.01+.49*torch.sigmoid(v[:,:,5]-1))[:,None,:]*eps[:,:,2][None])
        q = scale[:,None,None]*F.softplus(-.7+decay[None,None,:]*(inverse_softplus(radius0)[:,None,None]+.7)+v[:,:,6][:,None,:]+
             torch.tanh(v[:,:,7])[:,None,:]*common[None,:,None]+(.01+.49*torch.sigmoid(v[:,:,8]-1))[:,None,:]*eps[:,:,3][None])
        high=price*torch.exp((d+q).clamp(-8.,8.)); low=price*torch.exp((d-q).clamp(-8.,8.))
        ix=torch.tensor([h-1 for h in HORIZONS],device=x.device)
        return {'mean':p0[:,None]*torch.exp(meanlog.index_select(1,ix)), 'samples':summaries(price,high,low),
                'clip_count':(lp.abs()>8.).sum()+((d+q).abs()>8.).sum()+((d-q).abs()>8.).sum(),
                'daily':(price,high,low),'meanlog':meanlog,'variance':variance}


def make_model(family):
    if family=='gru_price': return PointGRU(True).double()
    if family=='gru_full': return PointGRU(False).double()
    if family=='free_path': return FreePath().double()
    if family in ('summary_state','state_price','state_marginal','state_step1'): return SummaryState().double()
    raise ValueError(family)


def empirical_energy(samples, truth, coordinate_mask=None):
    """Exact energy score of an equal-weight finite forecast distribution.

    Includes diagonal terms: a V statistic, not claimed an unbiased U statistic
    of the underlying continuous distribution. Gradients use the same measure.
    Sample axis is 1; coordinate mask is fixed by observation availability.
    """
    if coordinate_mask is None: coordinate_mask=torch.ones_like(truth)
    mask=coordinate_mask.to(truth.dtype)
    divisor=mask.sum(-1).clamp(min=1.).sqrt()
    a=((samples-truth[:,None,:])*mask[:,None,:]).square().sum(-1).clamp(min=1e-24).sqrt().mean(1)
    # cdist saves the four-dimensional pairwise-coordinate materialization.
    dist=torch.cdist(samples*mask[:,None,:],samples*mask[:,None,:],p=2)
    return (a-.5*dist.mean((1,2)))/divisor


def objective(output, y, valid, denominator, family, score_weight=.1):
    ix=slice(0,1) if family=='state_step1' else slice(None)
    mse=((output['mean'][:,ix]-y[:,ix,0])**2/denominator[ix,0]**2).mean()
    if output['samples'] is None: return mse, {'mse':float(mse.detach()),'energy':0.}
    sample=output['samples'][:,:,ix,:]/denominator[ix,:][None,None]
    truth=y[:,ix,:]/denominator[ix,:][None]
    mask=valid[:,ix,:]
    if family=='state_price':
        es=empirical_energy(sample[:,:,:,0],truth[:,:,0]).mean()
    elif family=='state_marginal':
        es=torch.stack([empirical_energy(sample[:,:,j,:],truth[:,j,:],mask[:,j,:]) for j in range(sample.shape[2])],1).mean()
    else:
        es=empirical_energy(sample.flatten(2),truth.flatten(1),mask.flatten(1)).mean()
    return mse+score_weight*es,{'mse':float(mse.detach()),'energy':float(es.detach())}
