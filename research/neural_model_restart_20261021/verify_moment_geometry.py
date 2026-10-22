# -*- coding: utf-8 -*-
"""Independent algebra check for the proposed hurdle moment parameterization.
Does not establish forecasting performance or economic variance invariance.
"""
import json
from pathlib import Path
import numpy as np


def moments(p,w,mp,mn,sp,sn):
    markmean=w*mp+(1-w)*mn
    markvar=w*sp**2+(1-w)*sn**2+w*(1-w)*(mp-mn)**2
    return p*markmean,p*markvar+p*(1-p)*markmean**2


def shape_to_components(p,mu,budget,w,d,a):
    sp,sn=np.exp(a),np.exp(-a)
    center=(2*w-1)*d
    variance=w*sp**2+(1-w)*sn**2+4*w*(1-w)*d**2
    multiplier=budget/np.sqrt(p*variance)
    return mu/p+multiplier*(d-center),mu/p+multiplier*(-d-center),multiplier*sp,multiplier*sn


def main():
    rng=np.random.default_rng(122)
    p0=rng.uniform(.3,.98,1000)
    w=rng.uniform(.1,.9,1000)
    mp,mn=np.exp(rng.normal(-1,.4,1000)),-np.exp(rng.normal(-1,.4,1000))
    sp,sn=np.exp(rng.normal(-.4,.5,1000)),np.exp(rng.normal(-.4,.5,1000))
    markmean=w*mp+(1-w)*mn
    markvar=w*sp**2+(1-w)*sn**2+w*(1-w)*(mp-mn)**2
    mu=p0*markmean
    budget=np.sqrt(p0*markvar)
    reference=np.sqrt(sp*sn)
    d=(mp-mn)/(2*reference)
    a=.5*np.log(sp/sn)
    restored=shape_to_components(p0,mu,budget,w,d,a)
    restore_error=max(float(np.abs(x-y).max()) for x,y in zip(restored,[mp,mn,sp,sn]))
    assert restore_error<1e-10
    p=rng.uniform(.05,.999,1000)
    params=shape_to_components(p,mu,budget,w,d,a)
    m,v=moments(p,w,*params)
    mean_error=float(np.abs(m-mu).max())
    variance_error=float(np.abs(v-(budget**2+(1-p)*mu**2/p)).max())
    assert mean_error<1e-10 and variance_error<1e-10
    # A numerical illustration, not evidence about future carbon-price risk.
    pp=.61;pp_new=.92;ww=.47;mplus=.74;mminus=-.68;splus=.41;sminus=.75
    orig_mean,orig_var=moments(pp,ww,mplus,mminus,splus,sminus)
    direct_mean,direct_var=moments(pp_new,ww,mplus,mminus,splus,sminus)
    mm=ww*mplus+(1-ww)*mminus
    vv=ww*splus**2+(1-ww)*sminus**2+ww*(1-ww)*(mplus-mminus)**2
    ss=np.sqrt(pp*vv);scale=np.sqrt(splus*sminus)
    dd=(mplus-mminus)/(2*scale);aa=.5*np.log(splus/sminus)
    coupled=shape_to_components(pp_new,orig_mean,ss,ww,dd,aa)
    coupled_mean,coupled_var=moments(pp_new,ww,*coupled)
    result={'random_cases':1000,'restore_component_max_error':restore_error,
            'marginal_mean_max_error':mean_error,'variance_identity_max_error':variance_error,
            'distribution_family_unchanged':True,
            'interpretation':'reparameterization and testable inductive constraint; not a law that future variance stays fixed',
            'illustration':{'p_old':pp,'p_new':pp_new,'old_mean':orig_mean,'old_variance':orig_var,
                             'fixed_mark_mean':direct_mean,'fixed_mark_variance':direct_var,
                             'moment_coupled_mean':coupled_mean,'moment_coupled_variance':coupled_var}}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
