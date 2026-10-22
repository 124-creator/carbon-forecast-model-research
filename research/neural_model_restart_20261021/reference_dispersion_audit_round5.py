"""Is the reference law mis-dispersed, and is the correction estimable pre-cut?

For each market x horizon: compare the dispersion of the frozen reference law
(spread of its quantile function about its median) with realised dispersion on
the pre-cut validation window, then apply that single estimated multiplier to the
test quantiles. No test information enters the multiplier.
"""
import sys,json
sys.path.insert(0,r'research/neural_model_restart_20261021')
import numpy as np,pandas as pd,torch
import forest_prior_round5 as FP
import forest_transport_round5 as F
T=FP.T;P=FP.P;R=F.R
TAUS=np.array(T.B.TAUS)

def pin(y,q):
    e=y[:,None]-q
    return float(2*np.maximum(TAUS*e,(TAUS-1)*e).mean())

def main():
    panel=pd.read_csv(T.B.PANEL,parse_dates=['date'])
    z=np.load(P/'market_info_round5_predictions.npz')
    base=np.load(P/'qrf_round4_probability_control_predictions.npz')
    rows=[]
    for market in T.B.CUT:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        D=T.add_context(d,T.E.build(d));D['market']=market
        cache=FP.load_cache(market,'final',D)
        sid_of=cache['sid'].numpy()
        for hi,h in enumerate(T.HOR):
            h=int(h);bank=cache['banks'][hi]
            for part in ['va','rf','te']:
                idx=D[part];idx=idx[D['hi'][idx]==hi]
                if len(idx)==0:continue
                sid=sid_of[idx];sc=D['scale'][idx]
                w=torch.tensor(bank['u'].numpy()[sid],dtype=torch.float64);w=w/w.sum(1,keepdim=True).clamp_min(1e-30)
                local=torch.tensor(D['local'][idx],dtype=torch.float64)
                p=.8*local+.2*float(bank['p'])
                op=dict(w=w,p=p,a=torch.ones(len(idx),dtype=torch.float64),b=torch.zeros(len(idx),dtype=torch.float64))
                q,_,_,_,_=R.mixture(torch.zeros(len(idx),dtype=torch.float64),bank['z'].double(),[op])
                Q=q.numpy()*sc[:,None];y=D['y'][idx];med=np.median(Q,axis=1)
                sp_q=np.std(Q-med[:,None],axis=1);sp_y=np.abs(y-med)
                rows.append(dict(market=market,h=h,part=part,n=len(idx),
                                 ratio=float(np.mean(sp_y**2)/np.mean(sp_q**2)),
                                 spread_q=float(np.mean(sp_q)),mae_med=float(np.mean(sp_y))))
    df=pd.DataFrame(rows)
    print(df.pivot_table(index=['market','h'],columns='part',values='ratio').round(3).to_string(),flush=True)
    out={'ratios':df.to_dict('records')}
    # apply the va-estimated multiplier to the test quantiles
    cells={}
    for market in T.B.CUT:
        Q=z[market+'_base_Q'];y=z[market+'_target'];h=z[market+'_horizon']
        rule=base[market+'_qrf_rule_atom_Q'];med=Q[:,7:8]
        c=np.ones(4)
        for hi,hh in enumerate(T.HOR):
            sel=df[(df.market==market)&(df.h==int(hh))&(df.part=='va')]
            c[hi]=np.sqrt(float(sel['ratio'].iloc[0])) if len(sel) else 1.0
        csel=np.array([c[list(T.HOR).index(int(x))] for x in h])
        Qc=med+csel[:,None]*(Q-med)
        for hh in T.HOR:
            k=h==hh
            cells[(market,int(hh))]=(pin(y[k],Qc[k])/pin(y[k],rule[k]),pin(y[k],Qc[k])/pin(y[k],Q[k]))
    r=np.array([v[0] for v in cells.values()]);g=100*(1-np.exp(np.log(r).mean()))
    r2=np.array([v[1] for v in cells.values()]);g2=100*(1-np.exp(np.log(r2).mean()))
    byh={int(hh):round(float(100*(1-np.exp(np.mean([np.log(cells[(m,hh)][0]) for m in T.B.CUT])))),2) for hh in T.HOR}
    bym={m:round(float(100*(1-np.exp(np.mean([np.log(cells[(m,hh)][0]) for hh in T.HOR])))),2) for m in T.B.CUT}
    print('pre-cut estimated multiplier: vs rule %+.3f%% vs base %+.3f%% wins %d/20'%(g,g2,int((r<1).sum())),flush=True)
    print('by h',byh,'by market',bym,flush=True)
    out['applied']=dict(vs_rule=g,vs_base=g2,wins=int((r<1).sum()),by_horizon=byh,by_market=bym,
                        multipliers={m:[float(x) for x in c] for m in ['shared']})
    (P/'reference_dispersion_audit_round5.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
