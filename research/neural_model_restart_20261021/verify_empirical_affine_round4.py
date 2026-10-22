# -*- coding: utf-8 -*-
"""Independent geometry check for conditional empirical location-scale transport.
This checks algebra and gradients, not predictive value or novelty.
"""
import json
from pathlib import Path
import numpy as np
import torch


def transport(z,logits,allowed,delta,ell):
    w=logits.masked_fill(~allowed,-1e9).softmax(1)
    u=allowed.to(logits.dtype)/allowed.sum(1,keepdim=True)
    m0=(u*z).sum(1);s0=torch.sqrt((u*(z-m0[:,None]).square()).sum(1))
    mw=(w*z).sum(1);sw=torch.sqrt((w*(z-mw[:,None]).square()).sum(1))
    mt=m0+s0*delta;st=s0*torch.exp(ell)
    a=st/sw;b=mt-a*mw
    return w,a,b,mt,st


def analytic(y,p,w,z,a,b):
    v=a[:,None]*z+b[:,None]
    cw=w.cumsum(1)-w;ca=(w*z).cumsum(1)-w*z
    half=(w*(z*cw-ca)).sum(1)
    return ((1-p)*y.abs()+p*(w*(v-y[:,None]).abs()).sum(1)
            -p.square()*a*half-p*(1-p)*(w*v.abs()).sum(1))


def cdf_integral(y,values,weights):
    # Include the observation in the partition; discrete CDF is constant in
    # every open interval and can be evaluated at its midpoint independently.
    ans=[]
    for yy,xx,ww in zip(y,values,weights):
        edges=np.sort(np.r_[xx,yy]);width=np.diff(edges);mid=(edges[:-1]+edges[1:])/2
        cdf=(ww[None,:]*(xx[None,:]<=mid[:,None])).sum(1)
        ans.append(np.sum(width*(cdf-(yy<=mid))**2))
    return np.array(ans)


def main():
    torch.set_num_threads(1);torch.manual_seed(701)
    dtype=torch.float64
    z=torch.tensor([-2.,-.9,-.35,.2,.7,1.8,3.2],dtype=dtype)
    logits=.2*torch.randn((7,7),dtype=dtype)
    allowed=torch.ones((7,7),dtype=torch.bool);allowed[1:4,4:]=False
    delta=torch.linspace(-.25,.3,7,dtype=dtype);ell=torch.linspace(-.2,.2,7,dtype=dtype)
    p=torch.tensor([0.,.2,.5,.7,.85,.98,1.],dtype=dtype)
    y=torch.tensor([-3.,-.3,0.,.431,1.1,2.,5.],dtype=dtype)
    w,a,b,mt,st=transport(z,logits,allowed,delta,ell)
    v=a[:,None]*z+b[:,None]
    mean=(w*v).sum(1);var=(w*(v-mean[:,None]).square()).sum(1)
    me=float((mean-mt).abs().max());ve=float((var-st.square()).abs().max())
    assert me<1e-12 and ve<1e-12
    values=torch.cat([v,torch.zeros((7,1),dtype=dtype)],1)
    weights=torch.cat([p[:,None]*w,(1-p)[:,None]],1)
    score=analytic(y,p,w,z,a,b)
    independent=cdf_integral(y.numpy(),values.numpy(),weights.numpy())
    se=float(np.max(np.abs(score.numpy()-independent)));assert se<1e-12
    atomerr=float(abs(score[0]-y[0].abs()));assert atomerr<1e-12
    w0,a0,b0,_,_=transport(z,torch.zeros_like(logits),allowed,torch.zeros(7,dtype=dtype),torch.zeros(7,dtype=dtype))
    ie=float(torch.max(torch.abs((a0[:,None]*z+b0[:,None])-z)))
    assert ie<1e-12
    lg=torch.tensor([[.1,-.1,.2,.05,-.12,.06,.08]],dtype=dtype,requires_grad=True)
    dl=torch.tensor([.13],dtype=dtype,requires_grad=True)
    el=torch.tensor([-.09],dtype=dtype,requires_grad=True)
    def fun(lg,dl,el):
        ww,aa,bb,_,_=transport(z,lg,torch.ones_like(lg,dtype=torch.bool),dl,el)
        return analytic(torch.tensor([.431],dtype=dtype),torch.tensor([.83],dtype=dtype),ww,z,aa,bb)
    grad=torch.autograd.gradcheck(fun,(lg,dl,el),eps=1e-6,atol=1e-5,rtol=1e-4)
    # Mix two different transported supports; cross-seed terms are indispensable.
    w2,a2,b2,_,_=transport(z,-logits,allowed,-delta,ell*.7)
    v2=a2[:,None]*z+b2[:,None]
    vals=torch.cat([v,v2,torch.zeros((7,1),dtype=dtype)],1)
    wei=torch.cat([.5*p[:,None]*w,.5*p[:,None]*w2,(1-p)[:,None]],1)
    integral=cdf_integral(y.numpy(),vals.numpy(),wei.numpy())
    pair=(wei[:,:,None]*wei[:,None,:]*(vals[:,:,None]-vals[:,None,:]).abs()).sum((1,2))
    brute=(wei*(vals-y[:,None]).abs()).sum(1)-.5*pair
    mixerr=float(np.max(np.abs(integral-brute.numpy())));assert mixerr<1e-12
    mean_seed=.5*(score+analytic(y,p,w2,z,a2,b2))
    violation=float(torch.clamp(brute-mean_seed,min=0).max());assert violation<1e-12
    result={'conditional_mean_max_error':me,'conditional_variance_max_error':ve,
            'zerohead_restore_max_error':ie,'analytic_CRPS_vs_CDF_max_error':se,
            'pure_zero_atom_error':atomerr,'gradient_check':bool(grad),
            'different_support_seed_mixture_CRPS_max_error':mixerr,
            'mixture_CRPS_convexity_violation':violation,
            'predictive_performance_established':False}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
