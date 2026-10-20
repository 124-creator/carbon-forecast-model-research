"""Independent synthetic date availability checks for primary and strict-prior modes."""
import numpy as np,pandas as pd
from run_dual_price import prepare

if __name__=='__main__':
    days=pd.date_range('2020-01-01',periods=90,freq='B')
    refdays=pd.date_range('2019-12-15',periods=115,freq='B')
    p=pd.DataFrame({'date':days,'price':40+np.cumsum(np.sin(np.arange(len(days))/5))})
    p['cover_ratio']=2+np.sin(np.arange(len(p))/20)*.2
    p['bidder_count']=20+np.arange(len(p))%4
    q=pd.DataFrame({'date':refdays,'price':40+np.cumsum(np.cos(np.arange(len(refdays))/5))})
    at=prepare(p,q,False);before=prepare(p,q,True)
    t=60
    for x,mode in [(at,False),(before,True)]:
        j=np.flatnonzero(x['origin']==t)[0];sourceidx=x['asof_idx'][t]
        if sourceidx>=0:
            assert (x['ref_days'][sourceidx] <= x['auction_days'][t]) if not mode else (x['ref_days'][sourceidx] < x['auction_days'][t])
        changed=p.copy();changed.loc[t+1:,'price']*=10
        changedq=q.copy();changedq.loc[changedq.date>p.date.iloc[t],'price']*=100
        xx=prepare(changed,changedq,mode)
        assert abs(x['del'][j]-xx['del'][j])<1e-12
        assert np.array_equal(x['z'][j],xx['z'][j])
        assert abs(x['m5'][j]-xx['m5'][j])<1e-12
        assert abs(x['m20'][j]-xx['m20'][j])<1e-12
        assert x['anchor'][j]==xx['anchor'][j] and x['truth'][j]!=xx['truth'][j]
    print('PASS_SYNTHETIC_PRETEND_FUTURE_MUTATION both modes')
