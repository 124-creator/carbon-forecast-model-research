"""Decompose the pooled-reference gain: same-date info vs unconditional shrinkage vs placebo dates."""
import sys,json
sys.path.insert(0,r'research/neural_model_restart_20261021')
import numpy as np,pandas as pd
import forest_prior_round5 as FP
T=FP.T
P=FP.P
TAUS=np.array(T.B.TAUS)
LAMS=[0.2,0.4,0.6]
MODES=['date','mean','shift1','shift5','shuffle']

def pinball(y,q):
    e=y[:,None]-q
    return float(2*np.maximum(TAUS*e,(TAUS-1)*e).mean())

def load():
    z=np.load(P/'market_info_round5_predictions.npz')
    panel=pd.read_csv(T.B.PANEL,parse_dates=['date'])
    data={}
    for market in T.B.CUT:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        dates=d['date'].values
        Q=z[market+'_base_Q'];origin=z[market+'_origin'];h=z[market+'_horizon'];y=z[market+'_target']
        rows={};pos={}
        for j in range(len(Q)):
            hh=int(h[j]);rows.setdefault(hh,{})[dates[origin[j]]]=j
            pos.setdefault(hh,[]).append(j)
        for hh in T.HOR:pos[hh]=sorted(pos[hh],key=lambda j:dates[origin[j]])
        data[market]=dict(Q=Q,h=h,y=y,rows=rows,pos=pos,
                          mean={int(hh):Q[h==hh].mean(0) for hh in T.HOR},
                          dates=dates,origin=origin)
    return z,data

def build_pool(market,data,mode):
    d=data[market];Q=d['Q'];h=d['h'];pool=np.zeros_like(Q)
    for j in range(len(Q)):
        hh=int(h[j]);dt=d['dates'][d['origin'][j]]
        acc=[];w=[]
        for other in T.B.CUT:
            if other==market:continue
            o=data[other]
            k=None
            if mode=='date':k=o['rows'].get(hh,{}).get(dt)
            elif mode=='mean':k=None
            elif mode=='shift1':
                seq=o['pos'][hh];dd=[o['dates'][o['origin'][q]] for q in seq]
                p=np.searchsorted(dd,dt)
                if 0<p<len(seq):k=seq[p-1]
                elif p>0:k=seq[p-1]
            elif mode=='shift5':
                seq=o['pos'][hh];dd=[o['dates'][o['origin'][q]] for q in seq]
                p=np.searchsorted(dd,dt)
                if p-5>=0:k=seq[p-5]
            elif mode=='shuffle':
                seq=o['pos'][hh];p=np.searchsorted([o['dates'][o['origin'][q]] for q in seq],dt)
                if p<len(seq):k=seq[min(p+97,len(seq)-1)]
            if k is None:
                acc.append(o['mean'][hh]);w.append(.5)
            else:
                acc.append(o['Q'][k]);w.append(1.0)
        pool[j]=np.average(np.array(acc),axis=0,weights=np.array(w))
    return pool

def main():
    z,data=load()
    out={}
    for mode in MODES:
        pools={m:build_pool(m,data,mode) for m in T.B.CUT}
        for lam in LAMS:
            cells={}
            for market in T.B.CUT:
                d=data[market];Q=d['Q'];h=d['h'];y=d['y']
                Qm=(1-lam)*Q+lam*pools[market]
                for hh in T.HOR:
                    msk=h==hh
                    cells[(market,int(hh))]=(pinball(y[msk],Qm[msk])/pinball(y[msk],Q[msk]))
            r=np.array(list(cells.values()))
            gain=float(100*(1-np.exp(np.log(r).mean())))
            by_h={int(hh):float(100*(1-np.exp(np.mean([np.log(cells[(m,hh)]) for m in T.B.CUT])))) for hh in T.HOR}
            per_m={m:float(100*(1-np.exp(np.mean([np.log(cells[(m,hh)]) for hh in T.HOR])))) for m in T.B.CUT}
            out[f'{mode}_{lam}']=dict(gain=gain,wins=int((r<1).sum()),by_horizon=by_h,per_market=per_m)
            print('%-8s lam %.1f  gain %+.3f%%  wins %2d/20  h %s  m %s'%(mode,lam,gain,int((r<1).sum()),
                  {k:round(v,2) for k,v in by_h.items()},{k:round(v,2) for k,v in per_m.items()}),flush=True)
    (P/'pooled_reference_modes_round5.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
