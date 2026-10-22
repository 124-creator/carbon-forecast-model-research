"""Probe: is the frozen reference the average of per-seed quantiles, or the quantile of the seed-averaged law?"""
import sys,json
sys.path.insert(0,r'research/neural_model_restart_20261021')
import numpy as np,pandas as pd,torch
import forest_prior_round5 as FP
import forest_transport_round5 as F
T=FP.T;Q=FP.Q;R=F.R
SEEDS=[11,22,33,44,55]
panel=pd.read_csv(T.B.PANEL,parse_dates=['date'])
d=panel[panel.market=='CEA'].sort_values('date').reset_index(drop=True)
D=T.add_context(d,T.E.build(d));D['market']='CEA'
cache=FP.load_cache('CEA','final',D)
old=np.load(FP.P/'forest_transport_round5_predictions.npz')
base=json.loads((FP.P/'qrf_round4.json').read_text(encoding='utf-8'))
fit=D['rf'];X,_=T.prepare(D,fit,True);xx=X['x'].numpy()
times,_,inv=np.unique(D['t'],return_index=True,return_inverse=True)
te=D['te']
for hi in [0,1,3]:
    h=int(T.HOR[hi]);bank=cache['banks'][hi]
    fit_h=fit[D['hi'][fit]==hi];nz=FP.nonzero_ids(D,fit,hi)
    lookup={int(i):j for j,i in enumerate(fit_h)};cols=np.array([lookup[int(i)] for i in nz])
    qr=np.flatnonzero(D['hi']==hi);qs=inv[qr]
    valid=np.zeros((len(times),4),dtype=bool);valid[inv,D['hi']]=True
    allowed=bank['mature'].numpy()[None,:]<=times[:,None]
    leaf=int(base['markets']['CEA']['selection'][str(h)]['chosen_minleaf'])
    ix=np.flatnonzero(D['hi'][te]==hi);sid=inv[D['t'][te][ix]]
    local=torch.tensor(D['local'][te][ix],dtype=torch.float64)
    p=.8*local+.2*float(bank['p'])
    zq=torch.tensor(D['z'][te][ix],dtype=torch.float64);zb=bank['z'].double();sc=D['scale'][te][ix]
    raws=[]
    for seed in SEEDS:
        f=FP.RandomForestRegressor(min_samples_leaf=leaf,random_state=seed,**Q.RF_PARAMS)
        f.fit(xx[fit_h],D['z'][fit_h]);ww,_=Q.leaf_weights(f,xx[fit_h],xx[qr],sanity=False)
        r=np.zeros((len(times),len(nz)),dtype=np.float32);r[qs]+=ww[:,cols];raws.append(r)
    def qof(u):
        w=torch.tensor(u[sid],dtype=torch.float64);w=w/w.sum(1,keepdim=True).clamp_min(1e-30)
        a1=torch.ones(len(ix),dtype=torch.float64);b0=torch.zeros(len(ix),dtype=torch.float64)
        q,c,pe,_,_=R.mixture(zq,zb,[dict(w=w,p=p,a=a1,b=b0)])
        return q.numpy()*sc[:,None]
    avgraw=np.mean(raws,axis=0)
    u_avg,_,_,_,_=FP.make_reference(avgraw,allowed,valid[:,hi])
    Q_avglaw=qof(u_avg)
    Qs=[]
    for r in raws:
        u,_,_,_,_=FP.make_reference(r,allowed,valid[:,hi])
        Qs.append(qof(u))
    Q_avgq=np.mean(Qs,axis=0)
    ref=old['CEA_forest_reference_only_Q'][ix]
    print('h',h,'quantile-of-averaged-law',f'{np.abs(Q_avglaw-ref).max():.4f}',
          'average-of-quantiles',f'{np.abs(Q_avgq-ref).max():.4f}')
