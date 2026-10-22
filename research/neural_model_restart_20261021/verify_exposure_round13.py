"""Independent numerical and information-set checks for exposure_round13."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr
import torch
import exposure_round13 as E


def main():
    torch.set_num_threads(1)
    net=E.ExposureNet(11,'observation_gate').double()
    cp,_=net(torch.zeros(1,11,dtype=torch.float64),torch.tensor([1.2],dtype=torch.float64),
             torch.tensor([.35],dtype=torch.float64),torch.tensor([3.],dtype=torch.float64),
             torch.tensor([5.],dtype=torch.float64))
    y=.4
    value=E.crps(torch.tensor([y],dtype=torch.float64),cp).sum()
    w,m,s=[v.detach().numpy()[0] for v in cp]
    assert np.allclose(w.sum(),1,atol=1e-12)
    def F(x):
        c=np.where(s>0,ndtr((x-m)/np.maximum(s,1e-100)),x>=m)
        return float(w@c)
    def f(x):return (F(x)-float(x>=y))**2
    integ=sum(quad(f,a,b,epsabs=1e-9)[0] for a,b in [(-np.inf,0),(0,y),(y,np.inf)])
    assert abs(integ-float(value.detach()))<2e-7
    value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    q=E.quantiles(tuple(v.detach().numpy() for v in cp))[0]
    for tau,qq in zip(E.TAUS,q):
        assert F(qq+1e-7)>=tau-1e-6 and F(qq-1e-7)<=tau+1e-6
    panel=pd.read_csv(E.PANEL,parse_dates=['date'])
    checked=[]
    for market in E.CUT:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        D=E.build(d); full=E.daily_features(d)
        for t in [D['val'],D['cut'],min(D['cut']+50,len(d)-1)]:
            prefix=E.daily_features(d.iloc[:t+1].copy())
            for key in ['x','age','base','prior','scale']:
                np.testing.assert_allclose(prefix[key][-1],full[key][t],rtol=1e-12,atol=1e-12)
        checked.append(market)
    out=dict(source_sha256=hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(),
             closed_form_CRPS=float(value.detach()),CDF_integral=integ,gradient_finite=True,
             quantile_inverse_CDF=True,prefix_invariance_markets=checked,
             mature_train_val_refit_labels=True,checks=E.sanity())
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
