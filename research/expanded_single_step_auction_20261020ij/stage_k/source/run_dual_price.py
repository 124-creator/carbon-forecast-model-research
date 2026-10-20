#!/usr/bin/env python3
"""Stage-k dual-source EUA next auction price, honest conditional/as-of-date experiment.
Primary hypothesis is a bounded context-conditioned spread correction versus
constant-spread, same-input linear and nonlinear controls. No new holdout claims.
"""
from __future__ import annotations
import argparse,zipfile,io,json,hashlib,math,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.optimize import minimize

P_ARCHIVE='local_only/inputs/EEX_EUA_T3PA_daily_development.csv'
Q_ARCHIVE='inputs/development/EUA_LONG_development.csv'
P_SHA='5a0c7580edebc2a7a5e66d7487568f6d99a8e08cca0785676c2aaeccad7db6bb'
Q_SHA='142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f'
YEARS=range(2018,2024)
ALPHAS=(.0001,.001,.01,.1,1,10,100)
GAMMAS=(.001,.01,.1,1,10)
SHRINK=(0.,.25,.5,1.)
LAMBDA=(0.,.1,.25,.5,.75,1.)
MAIN='bounded_depth_conditioned_spread_gate'
ARMS=('rw','reference_direct','blend_constant','auction_price_only_ridge','additive_spread_ridge','spread_context_interaction_ridge','fixed_spread_gate','bounded_depth_conditioned_spread_gate','hgb_same_information')

def sha(b):return hashlib.sha256(b).hexdigest()
def writej(path,obj):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def source(archive1,archive2):
    with zipfile.ZipFile(archive1) as z:pb=z.read(P_ARCHIVE)
    with zipfile.ZipFile(archive2) as z:qb=z.read(Q_ARCHIVE)
    if sha(pb)!=P_SHA or sha(qb)!=Q_SHA:raise ValueError('Upstream source bytes changed')
    p=pd.read_csv(io.BytesIO(pb));q=pd.read_csv(io.BytesIO(qb));
    for d in [p,q]:
        d.date=pd.to_datetime(d.date)
        if d.date.isna().any() or d.date.duplicated().any() or not d.date.is_monotonic_increasing or not np.isfinite(d.price).all() or (d.price<=0).any():raise ValueError('Bad archive price/data identity')
    if not (p.contract=='T3PA').all() or p.shape[0]!=2255 or q.shape[0]!=2839:raise ValueError('Wrong product/sample identity')
    return p,q

def prepare(p,q,strict_prior=False):
    auction_days=p.date.to_numpy(dtype='datetime64[D]'); ref_days=q.date.to_numpy(dtype='datetime64[D]');
    inds=np.searchsorted(ref_days,auction_days,side='left' if strict_prior else 'right')-1
    available=inds>=0
    matched=np.empty(len(p));matched[~available]=np.nan;matched[available]=q.price.to_numpy(float)[inds[available]]
    age=np.full(len(p),99.,dtype=float);age[available]=(auction_days[available]-ref_days[inds[available]]).astype(int)
    available &= age<=7
    matched[~available]=np.nan
    price=p.price.to_numpy(float); diff=np.diff(price,prepend=price[0]);cover=np.log(p.cover_ratio.to_numpy(float));bidder=p.bidder_count.to_numpy(float)
    origin=np.arange(32,len(p)-1);gap_days=np.r_[0,np.diff(auction_days).astype(int)]
    m5=[];m20=[];Z=[];DELTA=[];presence=[]
    for t in origin:
        a=diff[max(0,t-19):t+1];b=diff[max(0,t-4):t+1]
        xcover=cover[t]-np.mean(cover[max(0,t-20):t])
        xbid=bidder[t]-np.mean(bidder[max(0,t-20):t])
        m5.append(np.mean(b));m20.append(np.mean(a));
        valid=available[t]
        DELTA.append(matched[t]-price[t] if valid else 0.)
        presence.append(int(valid))
        Z.append([xcover, np.tanh(diff[t]/2.),np.log((np.sqrt(np.mean(b*b))+.1)/(np.sqrt(np.mean(a*a))+.1)),xbid, min(age[t],7)/7 if valid else 1.])
    return {'price':price,'origin':origin,'auction_days':auction_days,'ref_days':ref_days,'asof_idx':inds,
            'del':np.asarray(DELTA),'m5':np.asarray(m5),'m20':np.asarray(m20),'z':np.asarray(Z),
            'available':np.asarray(presence,dtype=bool),'ages':age[origin],
            'truth':price[origin+1],'anchor':price[origin], 'mode':'strict_prior' if strict_prior else 'asof_close'}

def scaled_feats(d,fit_mask):
    y0=d['truth']-d['anchor'];scale=max(float(np.sqrt(np.mean(y0[fit_mask]**2))),1e-8)
    S=(d['del']/scale)[:,None];M=np.column_stack((d['m5']/scale,d['m20']/scale))
    Z=StandardScaler().fit_transform(d['z'][fit_mask]) if False else None
    scaler=StandardScaler().fit(d['z'][fit_mask]);z=scaler.transform(d['z'])
    z=np.clip(z,-6,6)
    return {'scale':scale,'S':S,'M':M,'Z':z,'target':y0/scale,'presence':d['available'].astype(float)[:,None]}

def metrics(pp,obs,rw):
    mse=np.mean((pp-obs)**2);den=np.mean((rw-obs)**2)
    return {'MSE':float(mse),'RMSE':float(np.sqrt(mse)),'MAE':float(np.mean(np.abs(pp-obs))),
            'MAPE_percent':float(np.mean(np.abs(pp-obs)/np.abs(obs))*100),'R2_price':float(r2_score(obs,pp)),
            'Q_MSE_RW':float(mse/den)}

def ridge(f,y,train,alpha):
    # Columns follow a fixed, train-only standardized schema with positive ridge penalty.
    sc=StandardScaler().fit(f[train]);ft=sc.transform(f)
    m=Ridge(alpha=alpha*train.sum()).fit(ft[train],y[train])
    return m.predict(ft),{'coef':m.coef_.tolist(),'intercept':float(m.intercept_),'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist()}

def sigmoid(x):
    z=np.clip(x,-30,30);return 1/(1+np.exp(-z))

def fit_gate(D,M,Z,y,fit,gamma,conditioned):
    """Profiled bounded error-correction coefficient and linear trend on fit prefix.
    Theta includes intercept and 5 context coefficients. Fixed gate sets them=0.
    Analytical Jacobian; controls get the same optimizer iterations budget.
    """
    d=D[:,0];X=np.column_stack((np.ones(len(d)),M))
    dim=Z.shape[1] if conditioned else 0
    x=Z if conditioned else np.zeros((len(d),0))
    xx=np.column_stack((np.ones(len(d)),x))
    # Determine a valid train-only static spread coefficient initialization.
    den=float(np.sum(d[fit]**2)+1e-10)
    k=float(np.clip((d[fit]@y[fit])/den,.03,.97))
    starts=[np.r_[np.log(k/(1-k)),np.zeros(dim),np.zeros(3)],np.r_[0.,np.zeros(dim),np.zeros(3)]]
    def obj(w):
        theta=w[:1+dim];beta=w[1+dim:]
        rate=sigmoid(xx@theta);f=rate*d+X@beta;err=f[fit]-y[fit]
        penalty=gamma*(np.sum(theta[1:]**2)+np.sum(beta[1:]**2))
        loss=float(np.mean(err**2)+penalty)
        dtheta=((2/len(err))*(err*d[fit]*rate[fit]*(1-rate[fit])))@xx[fit]
        if len(dtheta)>1:dtheta[1:]+=2*gamma*theta[1:]
        dbeta=2/len(err)*(X[fit].T@err);dbeta[1:]+=2*gamma*beta[1:]
        return loss,np.r_[dtheta,dbeta]
    bound=[(-8,8)]+[(-6,6)]*dim+[(-5,5)]*3
    solutions=[minimize(obj,w,jac=True,method='L-BFGS-B',bounds=bound,options={'maxiter':400,'ftol':1e-10,'gtol':1e-7}) for w in starts]
    best=min(solutions,key=lambda res:(res.fun,res.nit))
    w=best.x;pred=sigmoid(xx@w[:1+dim])*d+X@w[1+dim:]
    if not np.isfinite(pred).all():raise FloatingPointError('bad finite gate')
    return pred,{'objective':float(best.fun),'success':bool(best.success),'iterations':int(best.nit),'coefficients':w.tolist(),
                 'gamma':gamma,'conditioned':conditioned,'avg_rate_fit':float(np.mean(sigmoid(xx[fit]@w[:1+dim]))),'static_initial_k':k,
                 'gradient_norm':float(np.max(np.abs(obj(w)[1])))}

def run_year(d,year,out):
    dates=d['auction_days'];o=d['origin'];b=np.searchsorted(dates,np.datetime64(f'{year}-01-01'));e=np.searchsorted(dates,np.datetime64(f'{year+1}-01-01'))
    a=b-100;tr=o+1<a;val=(o>=a)&(o+1<b);fi=o+1<b;test=(o>=b)&(o+1<e)
    if tr.sum()<500 or val.sum()<90 or test.sum()<180:raise ValueError('not enough training or valid events '+str(year))
    s=scaled_feats(d,tr);target=s['target'];D=s['S'];M=s['M'];Z=s['Z'];P=s['presence'];
    linear={
     'auction_price_only_ridge':M,
     'additive_spread_ridge':np.column_stack((D,M,P,Z)),
     'spread_context_interaction_ridge':np.column_stack((D,M,P,Z,D*Z)),
     'hgb_same_information':np.column_stack((D,M,P,Z))}
    pred={'rw':np.zeros_like(target),'reference_direct':D[:,0]}
    records={'year':year,'mode':d['mode'],'n_train':int(tr.sum()),'n_val':int(val.sum()),'n_test':int(test.sum()),
             'source_quote_available_share_test':float(P[test].mean()),'source_quote_age_days_mean_test':float(d['ages'][test].mean()),
             'source_quote_age_days_max_test':float(d['ages'][test].max()),'scale_training_only':s['scale'],
             'last_validation_label_date':str(dates[o[val][-1]+1]),'first_test_origin_date':str(dates[o[test][0]]),
             'selection':{}}
    # Do not tune reference_direct: its meaning is literally the last quoted price.
    lbest=min([(float(np.mean((k*D[val,0]-target[val])**2)),k) for k in LAMBDA],key=lambda t:t)
    pred['blend_constant']=lbest[1]*D[:,0]
    records['selection']['blend_constant']={'blend_weight':lbest[1],'validation_loss':lbest[0],'tested_weights':list(LAMBDA)}
    for name,features in linear.items():
        if name=='hgb_same_information':
            candidates=[]
            for leaf in (7,15):
                for l2 in (.1,1.):
                    model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=leaf,l2_regularization=l2,learning_rate=.05,random_state=17,early_stopping=False)
                    model.fit(features[tr],target[tr]);q=model.predict(features[val])
                    for shrink in SHRINK:candidates.append((float(np.mean((shrink*q-target[val])**2)),leaf,l2,shrink))
            choice=min(candidates)
            _,leaf,l2,shrink=choice
            model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=leaf,l2_regularization=l2,learning_rate=.05,random_state=17,early_stopping=False)
            model.fit(features[fi],target[fi]);base=model.predict(features);records['selection'][name]={'choice':choice,'trials_count':len(candidates)}
        else:
            cand=[]
            for alpha in ALPHAS:
                q,_=ridge(features,target,tr,alpha)
                for shrink in SHRINK:cand.append((float(np.mean((shrink*q[val]-target[val])**2)),alpha,shrink))
            choice=min(cand);_,alpha,shrink=choice;base,_=ridge(features,target,fi,alpha)
            records['selection'][name]={'alpha_mean':alpha,'shrink':shrink,'validation_loss':choice[0],'trials_count':len(cand)}
        pred[name]=shrink*base;pred[name+'_unshrunk']=base
    for name,conditioned in [('fixed_spread_gate',False),('bounded_depth_conditioned_spread_gate',True)]:
        cand=[]
        for gamma in GAMMAS:
            v,m=fit_gate(D,M,Z,target,tr,gamma,conditioned)
            for shrink in SHRINK:cand.append((float(np.mean((shrink*v[val]-target[val])**2)),gamma,shrink))
        choice=min(cand);_,gamma,shrink=choice
        v,m=fit_gate(D,M,Z,target,fi,gamma,conditioned)
        pred[name]=shrink*v;pred[name+'_unshrunk']=v
        records['selection'][name]={'gamma':gamma,'shrink':shrink,'validation_loss':choice[0],'trials_count':len(cand),'final_fit':m}
    obs=d['truth'][test];rw=d['anchor'][test];forecast={k:rw+s['scale']*v[test] for k,v in pred.items()}
    scored={k:metrics(v,obs,rw) for k,v in forecast.items()}
    (out/'private').mkdir(exist_ok=True,parents=True)
    (out/'cells').mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'private'/f'{d["mode"]}_{year}.npz',observed=obs,rw=rw,origins=o[test],**{k:v for k,v in forecast.items() if k!='rw'})
    records['scores']=scored;writej(out/'cells'/f'{d["mode"]}_{year}.json',records)
    return {'mode':d['mode'],'year':year,'n':int(test.sum()),'scores':scored,'source_available':records['source_quote_available_share_test'],
            'source_age_mean':records['source_quote_age_days_mean_test']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary-archive',required=True);ap.add_argument('--aux-archive',required=True);ap.add_argument('--out',required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    p,q=source(args.primary_archive,args.aux_archive)
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    origin_hash=sha(Path(__file__).read_bytes())
    all_data={mode:prepare(p,q,strict_prior=(mode=='strict_prior')) for mode in ('asof_close','strict_prior')}
    proto={'stage':'k','candidate':MAIN,'source_primary_sha':P_SHA,'source_aux_sha':Q_SHA,'source_code_sha':origin_hash,
           'primary_count':len(p),'aux_count':len(q),'forecast_target':'next EEX auction price, not same-day clearing',
           'information_conditions':['end_of_day_at_or_before_current_auction_date','strictly_prior_aux_quote_date'],
           'date_identification':'author auxiliary publication time and precise instrument not established; conditional development only',
           'years':list(YEARS),'alpha_mean':list(ALPHAS),'gammas':list(GAMMAS),'shrinks':list(SHRINK),
           'no_2024_or_2025_confirmation':True,'models':list(ARMS)}
    writej(out/'PROTOCOL.json',proto)
    if args.preflight:
        for mode,data in all_data.items():
            for year in YEARS:
                o=data['origin'];dates=data['auction_days'];b=np.searchsorted(dates,np.datetime64(f'{year}-01-01'));e=np.searchsorted(dates,np.datetime64(f'{year+1}-01-01'))
                te=(o>=b)&(o+1<e)
                for i in o[te]:
                    s=data['asof_idx'][i]
                    if s>=0:
                        if mode=='asof_close':assert data['ref_days'][s]<=dates[i]
                        else:assert data['ref_days'][s]<dates[i]
                print('PREFLIGHT',mode,year,'N',te.sum(),'available',data['available'][te].mean(),'minage',min(data['ages'][te]))
        return
    records=[]
    for mode,data in all_data.items():
        for year in YEARS:
            r=run_year(data,year,out);records.append(r)
            print('DONE',mode,year,'Q',{k:round(r['scores'][k]['Q_MSE_RW'],4) for k in ARMS},flush=True)
    if sha(Path(__file__).read_bytes())!=origin_hash:raise ValueError('Source code mutated during fitting')
    writej(out/'COMPLETE.json',{'status':'COMPLETE','cells':len(records),'outcomes':records,'source_code_unchanged':True})
    for mode in ('asof_close','strict_prior'):
        pts=[x for x in records if x['mode']==mode]
        rank=sorted([(name,float(np.mean([z['scores'][name]['Q_MSE_RW'] for z in pts]))) for name in ARMS],key=lambda x:x[1]);print('RANKING',mode,rank)
if __name__=='__main__':main()
