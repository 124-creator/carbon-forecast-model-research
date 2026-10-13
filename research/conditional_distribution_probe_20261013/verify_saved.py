"""Replay saved predictions and independently score every saved metric."""
import json,sys
import numpy as np
from probe import OUT,read_prefix,configs,forecast,require,H,LO,HI,TAU,ALPHA,verify

def independent(q,y):
    e=y[:,:,None]-q
    pin=np.where(e>=0,TAU*e,(TAU-1)*e)
    low=q[:,:,LO]; high=q[:,:,HI]; w=high-low
    lt=y[:,:,None]<low; ut=y[:,:,None]>high
    score=w+np.where(lt,2/ALPHA*(low-y[:,:,None]),0)+np.where(ut,2/ALPHA*(y[:,:,None]-high),0)
    wis=(np.abs(y-q[:,:,4])*.5+np.sum(score*ALPHA/2,axis=2))/4.5
    return dict(n=len(y),wis_h=wis.mean(0).tolist(),pinball_h_tau=pin.mean(0).tolist(),width_h_interval=w.mean(0).tolist(),coverage_h_interval=(~lt&~ut).mean(0).tolist(),lower_tail_h_interval=lt.mean(0).tolist(),upper_tail_h_interval=ut.mean(0).tolist())
def compare(a,b):
    require(a['n']==b['n'],'score n')
    for key in a:
        if key!='n': require(np.allclose(a[key],b[key],atol=1e-12,rtol=1e-12),key)
def main():
    d,p,audit=read_prefix(); initial=int(np.sum(d<'2020-01-01')); cc=configs(); rr=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'))
    logs=json.loads((OUT/'MATURITY_LOGS_LOCAL_ONLY.json').read_text(encoding='utf-8'))['by_configuration']
    with np.load(OUT/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False) as z:
        oo=z['origins']; yy=z['true_changes']; qq=z['quantile_changes']; require(qq.shape==(12,451,3,9),'shape')
        require(np.array_equal(yy,p[oo[:,None]+H]-p[oo,None]),'targets')
        require(np.array_equal(z['quantile_prices'],qq+p[oo][None,:,None,None]),'prices')
        require(np.array_equal(z['interval_lower_prices'],qq[:,:,:,LO]+p[oo][None,:,None,None]),'low interval')
        require(np.array_equal(z['interval_upper_prices'],qq[:,:,:,HI]+p[oo][None,:,None,None]),'high interval')
    years=np.array([x[:4] for x in d[oo]])
    for k,c in enumerate(cc):
        replay,ll=forecast(p,initial,oo,c); require(np.array_equal(replay,qq[k]),'full replay'); require(ll==logs[k],'log replay')
        require(all(row[2]+row[1]==row[3] and row[3]<=row[0] for row in ll),'maturity')
        compare(independent(qq[k],yy),rr[k]['overall'])
        for yr in ['2020','2021']: compare(independent(qq[k,years==yr],yy[years==yr]),rr[k]['by_year'][yr])
    receipt=verify(d,p,initial,oo,qq,cc,yy); receipt.update(saved_metrics_independent=True,all_configuration_full_replay=True,maturity_logs_verified=True)
    (OUT/('SAVED_VERIFY_OPTIMIZED.json' if not __debug__ else 'SAVED_VERIFY.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8'); print(json.dumps(receipt))
if __name__=='__main__': main()
