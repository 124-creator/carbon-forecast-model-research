"""Probe: does cross-sectional pooling of the conditional reference quantiles help?

Uses the already-frozen test quantiles from market_info_round5_predictions.npz.
Lambda is swept on the test set and the whole curve is reported; this is a
screening probe with no selection claim (a real model would pick lambda on the
inner validation window).
"""
import sys,json
sys.path.insert(0,r'research/neural_model_restart_20261021')
import numpy as np,pandas as pd
import forest_prior_round5 as FP
T=FP.T
P=FP.P
TAUS=np.array(T.B.TAUS)
LAMS=[0.0,0.2,0.4,0.6,0.8,1.0]

def pinball(y,q):
    e=y[:,None]-q
    return float(2*np.maximum(TAUS*e,(TAUS-1)*e).mean())

def main():
    z=np.load(P/'market_info_round5_predictions.npz')
    panel=pd.read_csv(T.B.PANEL,parse_dates=['date'])
    dates={};data={}
    for market in T.B.CUT:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        dates[market]=d['date'].values
        Q=z[market+'_base_Q'];origin=z[market+'_origin'];h=z[market+'_horizon'];y=z[market+'_target']
        rows={}
        for j,hh in enumerate(h):
            rows.setdefault(int(hh),{})[dates[market][origin[j]]]=j
        data[market]=dict(Q=Q,h=h,y=y,rows=rows,
                          mean={int(hh):Q[h==hh].mean(0) for hh in T.HOR})
    out={}
    for lam in LAMS:
        cells={}
        for market in T.B.CUT:
            d=data[market];Q=d['Q'];h=d['h'];y=d['y']
            pool=np.zeros_like(Q)
            for j in range(len(Q)):
                hh=int(h[j]);dt=dates[market][z[market+'_origin'][j]]
                acc=[];w=[]
                for other in T.B.CUT:
                    if other==market:continue
                    k=data[other]['rows'].get(hh,{}).get(dt)
                    if k is None:
                        acc.append(data[other]['mean'][hh]);w.append(.5)
                    else:
                        acc.append(data[other]['Q'][k]);w.append(1.0)
                pool[j]=np.average(np.array(acc),axis=0,weights=np.array(w))
            Qm=(1-lam)*Q+lam*pool
            for hh in T.HOR:
                m=h==hh
                cells[(market,int(hh))]=dict(own=pinball(y[m],Q[m]),mix=pinball(y[m],Qm[m]))
        r=np.array([v['mix']/v['own'] for v in cells.values()])
        gain=float(100*(1-np.exp(np.log(r).mean())))
        by_h={int(hh):float(100*(1-np.exp(np.mean([np.log(cells[(m,hh)]['mix']/cells[(m,hh)]['own']) for m in T.B.CUT])))) for hh in T.HOR}
        per_m={m:float(100*(1-np.exp(np.mean([np.log(cells[(m,hh)]['mix']/cells[(m,hh)]['own']) for hh in T.HOR])))) for m in T.B.CUT}
        out[str(lam)]=dict(gain=gain,wins=int((r<1).sum()),by_horizon=by_h,per_market=per_m)
        print('lambda',lam,'gain %+.4f%%'%gain,'wins %d/20'%int((r<1).sum()),
              'h',{k:round(v,3) for k,v in by_h.items()},'m',{k:round(v,3) for k,v in per_m.items()},flush=True)
    (P/'pooled_reference_probe_round5.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
