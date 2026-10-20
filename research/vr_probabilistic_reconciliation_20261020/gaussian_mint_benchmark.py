#!/usr/bin/env python3
"""Gaussian MinT-like WLS projection baseline for h5/h20 carbon returns.
Post-result benchmark extension: frozen separately on GitHub before scoring.
"""
import hashlib,json,pathlib,zlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import pdist

BASE=pathlib.Path(__file__).parent
STORED=json.loads((BASE/'results'/'SUMMARY.json').read_text())
Q=np.array([.05,.1,.25,.5,.75,.9,.95]);N=252;SEED=20261020
A=np.array([[-1.,-1.,-1.,-1.,1.]])
I=np.eye(5)

def baseline_means(lp,t):
    out=[]
    for h in [5,20]:
        s=np.arange(t-h-N+1,t-h+1,dtype=int)
        assert s.min()>=0 and (s+h).max()<=t
        out.append((lp[s+h]-lp[s]).mean())
    return np.array([out[0]]*4+[out[1]])

def true_vector(lp,t):
    idx=np.arange(t,t+21,5)
    parts=np.diff(lp[idx])
    return np.r_[parts,parts.sum()]

def energy_s(s,y):
    x=np.asarray(s)
    return float(np.linalg.norm(x-y[None,:],axis=1).mean()-pdist(x).sum()/len(x)**2)

def metrics(y,qt):
    d=y[:,None]-qt
    pb=np.maximum(Q*d,(Q-1)*d).mean()
    return dict(pinball=float(pb),coverage80=float(np.mean((y>=qt[:,1])&(y<=qt[:,-2]))),
        coverage90=float(np.mean((y>=qt[:,0])&(y<=qt[:,-1]))),
        width80=float(np.mean(qt[:,-2]-qt[:,1])),width90=float(np.mean(qt[:,-1]-qt[:,0])))

def run():
    records=[];max_constraint=0.;
    for market in ['GUANGZHOU','HUBEI','SHANGHAI','EUA_LONG']:
        p=BASE/'private_inputs'/(market+'_development.csv')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==STORED['dataset_sha256'][market]
        df=pd.read_csv(p).sort_values('date');dt=pd.to_datetime(df.date).values.astype('datetime64[D]')
        lp=np.log(df.price.values.astype(float))
        for year in [2022,2023]:
            b=np.searchsorted(dt,np.datetime64(f'{year}-01-01'))
            e=np.searchsorted(dt,np.datetime64(f'{year+1}-01-01'))
            v=np.arange(b-100,b-20)
            tst=np.arange(b,e-20)
            assert len(v)==80 and len(tst)>=100 and max(v+20)<b
            errors=np.array([true_vector(lp,s)-baseline_means(lp,s) for s in v]);
            what=np.cov(errors,rowvar=False,ddof=1)
            what=(what+what.T)/2
            W=.5*what+.5*np.diag(np.diag(what))
            assert np.linalg.eigvalsh(W).min()>0
            den=(A@W@A.T).item()
            assert den>1e-14
            P=I-(W@A.T@A)/den
            Wrec=P@W@P.T
            Wrec=(Wrec+Wrec.T)/2
            max_constraint=max(max_constraint,float(np.max(np.abs(A@P))),float(np.max(np.abs(A@Wrec))))
            all_forecasts={k:{h:[] for h in [5,20]} for k in ['gaussian_unreconciled','gaussian_mint']}
            true={h:[] for h in [5,20]}
            es={k:[] for k in ['gaussian_unreconciled','gaussian_mint']}
            for t in tst:
                m=baseline_means(lp,int(t));mr=P@m
                y=true_vector(lp,int(t));
                assert abs(y[-1]-y[:4].sum())<1e-10
                for h,j in [(5,0),(20,4)]:
                    true[h].append(y[j])
                    all_forecasts['gaussian_unreconciled'][h].append(m[j]+norm.ppf(Q)*np.sqrt(W[j,j]))
                    all_forecasts['gaussian_mint'][h].append(mr[j]+norm.ppf(Q)*np.sqrt(max(Wrec[j,j],1e-15)))
                # Same base draws then exact WLS reconciled, no future observations.
                rng=np.random.default_rng(SEED+year*100000+int(t)+zlib.crc32(market.encode())%10000)
                z=rng.multivariate_normal(m,W,size=128)
                zz=z@P.T
                assert float(np.max(np.abs(zz@A.T)))<1e-10
                es['gaussian_unreconciled'].append(energy_s(z[:,[0,4]],y[[0,4]]))
                es['gaussian_mint'].append(energy_s(zz[:,[0,4]],y[[0,4]]))
            scored={name:{str(h):metrics(np.array(true[h]),np.array(all_forecasts[name][h])) for h in [5,20]} for name in es}
            for name in es:
                scored[name]['mean_pinball']=float(np.mean([scored[name][str(h)]['pinball'] for h in [5,20]]))
                scored[name]['energy']=float(np.mean(es[name]))
            records.append({'market':market,'year':year,'n_test':len(tst),'n_validation':len(v),
                'covariance_shrinkage':.5,'strict_validation_maturity':True,
                'min_eigenvalue_W':float(np.linalg.eigvalsh(W).min()),'metrics':scored})
            print('Gaussian',market,year,'unreconciled',round(scored['gaussian_unreconciled']['mean_pinball'],7),'mint',round(scored['gaussian_mint']['mean_pinball'],7),'Δ%',round(100*(1-scored['gaussian_mint']['mean_pinball']/scored['gaussian_unreconciled']['mean_pinball']),2),flush=True)
    agg={}
    for subset,rows in [('eight_cells',records),('six_chinese_cells',records[:6])]:
        agg[subset]={}
        for name in ['gaussian_unreconciled','gaussian_mint']:
            agg[subset][name]={k:float(np.mean([x['metrics'][name]['mean_pinball' if k=='mean_pinball' else 'energy'] for x in rows])) for k in ['mean_pinball','energy']}
        comparable=STORED['aggregates']['four_market_eight_cell' if subset=='eight_cells' else 'three_chinese_market_six_cell']
        agg[subset]['empirical_direct']={'mean_pinball':comparable['empirical_direct']['mean_pinball']}
        agg[subset]['selected_vr_mix']={'mean_pinball':comparable['selected_vr_mix']['mean_pinball']}
        agg[subset]['selected_sqrt_mix']={'mean_pinball':comparable['selected_sqrt_mix']['mean_pinball']}
    r={'status':'COMPLETE_POST_RESULT_EXPOSED_DEVELOPMENT','method_identity':'Gaussian_MinT_type_not_full_original_2024_IJF_reproduction',
      'distribution':'Normal','covariance_estimation':'80 historical fully-matured validation-origin errors','covariance_shrinkage':.5,
      'max_additive_constraint_residual':max_constraint,'cells':records,'aggregate':agg,
      'raw_prices_or_predictions_published':False}
    path=BASE/'results'/'GAUSSIAN_MIN_T_BENCHMARK.json'
    path.write_text(json.dumps(r,ensure_ascii=False,indent=2))
    print('FINAL',agg,'constraint',max_constraint)
if __name__=='__main__':run()
