#!/usr/bin/env python3
"""Stage-j: prespecified causal features for *next* EEX T3PA auction price.

Requires private source zip from author research archive (not redistributed).
Only past/current completed auctions available for each prediction origin.
All six older years are exposed development data, not an untouched holdout.
"""
from __future__ import annotations
import argparse, hashlib, io, json, zipfile, math, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

SOURCE_MEMBER='local_only/inputs/EEX_EUA_T3PA_daily_development.csv'
SOURCE_SHA='5a0c7580edebc2a7a5e66d7487568f6d99a8e08cca0785676c2aaeccad7db6bb'
YEARS=list(range(2018,2024)); LOOKBACK=32; VALIDATION_EVENTS=100
ALPHAS=(1e-4,1e-3,1e-2,.1,1,10,100)
SHRINK=(0.,.25,.5,1.)
MODELS=('price_response','signed_price_only','auction_additive','signed_coverage_interaction_ridge','price_ridge32','auction_ridge32','quadratic_context_ridge','auction_hgb','rw')

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def sha(b):return hashlib.sha256(b).hexdigest()

def open_verified(archive):
    with zipfile.ZipFile(archive) as z:
        b=z.read(SOURCE_MEMBER)
    if sha(b)!=SOURCE_SHA:raise ValueError('Private source hash changed: '+sha(b))
    df=pd.read_csv(io.BytesIO(b));df.date=pd.to_datetime(df.date)
    if df.date.duplicated().any() or not df.date.is_monotonic_increasing:raise ValueError('Source order/unique dates failed')
    if not (df.contract=='T3PA').all() or not np.isfinite(df.price).all() or (df.price<=0).any():raise ValueError('Contract/price identity failed')
    for col in ['auction_volume','total_bid_volume','cover_ratio','bidder_count','successful_bidder_count']:
        if not np.isfinite(df[col]).all() or (df[col]<=0).any():raise ValueError('Invalid source column: '+col)
    if not np.allclose(df.cover_ratio,df.total_bid_volume/df.auction_volume,rtol=1e-9,atol=1e-7):raise ValueError('Cover identity failed')
    return df

def signed(x):return np.column_stack((np.maximum(x,0),np.minimum(x,0)))

def build_features(d, start=LOOKBACK):
    p=d.price.to_numpy(float); r=np.diff(p,prepend=p[0]);dates=d.date.to_numpy(dtype='datetime64[D]')
    days=np.r_[0,np.diff(dates).astype(int)]
    cover=np.log(d.cover_ratio.to_numpy(float)); v=np.log(d.auction_volume.to_numpy(float)); bids=np.log(d.total_bid_volume.to_numpy(float));
    participation=d.bidder_count.to_numpy(float);success=d.successful_bidder_count.to_numpy(float)/participation
    if len(p)<=start+5:raise ValueError('Too short')
    origin=np.arange(start,len(p)-1)
    def trend(i,window):
        return np.mean(r[max(0,i-window+1):i+1])
    def std(i,window):
        a=r[max(0,i-window+1):i+1];return np.sqrt(np.mean(a*a))
    # No look-ahead: all indices for a forecast at t have i<=t.
    price_vec=[];depth_vec=[];lag_vec=[];base_vec=[];asym_vec=[];interaction_vec=[]
    for i in origin:
        hist=r[i-31:i+1]
        fast=np.dot(hist,(1-.5)*.5**np.arange(31,-1,-1))
        slow=np.dot(hist,(1-.9)*.9**np.arange(31,-1,-1))
        st=np.sqrt(np.mean(hist**2))
        pr=np.array([r[i],fast,slow,trend(i,5),trend(i,20),np.log((std(i,5)+.1)/(std(i,20)+.1))])
        past_cover=cover[max(0,i-20):i]
        cv=cover[i]-np.mean(past_cover)
        part=participation[i]-np.mean(participation[max(0,i-20):i])
        vk=v[i]-np.mean(v[max(0,i-20):i])
        weekday=d.date.iloc[i].weekday()
        # all auction quantities refer to t, not the next (unseen) auction t+1.
        depth=np.array([cover[i],cv,vk,bids[i]-v[i],np.log(participation[i]),part,success[i],np.log1p(days[i]),
                        math.sin(2*math.pi*weekday/7),math.cos(2*math.pi*weekday/7)])
        a=np.r_[signed(np.array([r[i]])).ravel(),signed(np.array([fast])).ravel(),signed(np.array([slow])).ravel()]
        # Main proposal: directional response gated by lagged realized bid-cover surprises.
        it=np.array([max(r[i],0)*cv,min(r[i],0)*cv,max(fast,0)*cv,min(fast,0)*cv,
                     max(slow,0)*cv,min(slow,0)*cv, fast*part, slow*part, fast*vk, slow*vk])
        price_vec.append(pr);depth_vec.append(depth);lag_vec.append(hist)
        asym_vec.append(a);interaction_vec.append(it)
    P=np.asarray(price_vec);D=np.asarray(depth_vec);L=np.asarray(lag_vec);A=np.asarray(asym_vec);I=np.asarray(interaction_vec)
    def pairs(X,Z):return np.einsum('ni,nj->nij',X,Z).reshape(len(X),-1)
    feats={
        'price_response':P,
        'signed_price_only':np.column_stack((P,A)),
        'auction_additive':np.column_stack((P,A,D)),
        'signed_coverage_interaction_ridge':np.column_stack((P,A,D,I)),
        'price_ridge32':L,
        'auction_ridge32':np.column_stack((L,D)),
        'quadratic_context_ridge':np.column_stack((P,A,D,pairs(P,A),pairs(P,D),pairs(A,D))),
        'auction_hgb':np.column_stack((L,D)),
    }
    for key,arr in feats.items():
        if not np.isfinite(arr).all():raise ValueError('Feature not finite '+key)
    return feats,origin,p[origin],p[origin+1],dates

def metrics(pred,obs,rw):
    if len(obs)!=len(pred) or len(obs)!=len(rw):raise ValueError('Length mismatch')
    e=np.asarray(pred)-obs; den=np.mean((rw-obs)**2); mse=np.mean(e*e)
    if den<=0:raise ValueError('Zero RW MSE')
    return {'n':len(e),'MSE':float(mse),'RMSE':float(np.sqrt(mse)),'MAE':float(np.mean(np.abs(e))),
            'MAPE_percent':float(100*np.mean(np.abs(e)/np.maximum(np.abs(obs),1e-8))),
            'R2_price':float(r2_score(obs,pred)),'Q_MSE_RW':float(mse/den)}

def fitting_ridge(feat,y,mask,alpha):
    scaler=StandardScaler().fit(feat[mask]); z=scaler.transform(feat)
    m=Ridge(alpha=float(alpha)*int(mask.sum())).fit(z[mask],y[mask]);return m.predict(z),{'coef':m.coef_.tolist(),'intercept':float(m.intercept_),
                                 'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()}

def run_cell(d,feats,origin,anchor,truth,dates,year,out):
    b=int(np.searchsorted(dates,np.datetime64(f'{year}-01-01')))
    e=int(np.searchsorted(dates,np.datetime64(f'{year+1}-01-01')))
    a=b-VALIDATION_EVENTS
    tr=(origin+1<a);va=(origin>=a)&(origin+1<b);fi=(origin+1<b);te=(origin>=b)&(origin+1<e)
    if tr.sum()<400 or va.sum()<60 or te.sum()<50:raise ValueError('Not enough valid events '+str(year))
    p=d.price.to_numpy(float);r=np.diff(p,prepend=p[0]);scale=np.sqrt(np.mean(r[1:a]**2));y=(truth-anchor)/scale
    # y has next price change but never enters candidate features.
    val_truth=y[va];test_truth=truth[te];test_rw=anchor[te];scores={'rw':metrics(test_rw,test_truth,test_rw)}
    rec={'year':year,'n_train':int(tr.sum()),'n_validation':int(va.sum()),'n_test':int(te.sum()),'last_train_target_date':str(dates[origin[tr][-1]+1]),
         'last_validation_target_date':str(dates[origin[va][-1]+1]), 'first_test_origin_date':str(dates[origin[te][0]]),
         'first_test_label_date':str(dates[origin[te][0]+1]),'last_test_label_date':str(dates[origin[te][-1]+1]),
         'scale_training_only':float(scale),'families':{}}
    private={'true':test_truth,'rw':test_rw,'origin_record_indices':origin[te]}
    for name in MODELS:
        if name=='rw':continue
        ft=feats[name]
        if name=='auction_hgb':
            best=None;trials=[]
            for leaves in (7,15):
                for l2 in (.1,1.):
                    m=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=leaves,l2_regularization=l2,learning_rate=.05,early_stopping=False,random_state=17).fit(ft[tr],y[tr]);vr=m.predict(ft[va]);
                    for sh in SHRINK:
                        loss=float(np.mean((sh*vr-val_truth)**2));choice=(loss,leaves,l2,sh)
                        trials.append({'leaf':leaves,'l2':l2,'shrink':sh,'validation_loss':loss})
                        if best is None or choice<best:best=choice
            _,leaves,l2,sh=best
            mod=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=leaves,l2_regularization=l2,learning_rate=.05,early_stopping=False,random_state=17).fit(ft[fi],y[fi])
            predict=mod.predict(ft)
            best_config={'leaf':leaves,'l2':l2,'shrink':sh}
        else:
            best=None;trials=[]
            for alpha in ALPHAS:
                pr,_=fitting_ridge(ft,y,tr,alpha)
                for sh in SHRINK:
                    loss=float(np.mean((sh*pr[va]-val_truth)**2));choice=(loss,alpha,sh)
                    trials.append({'alpha_mean':alpha,'shrink':sh,'validation_loss':loss})
                    if best is None or choice<best:best=choice
            _,alpha,sh=best
            predict,_=fitting_ridge(ft,y,fi,alpha)
            best_config={'alpha_mean':alpha,'shrink':sh}
        pred_unsh=anchor[te]+scale*predict[te]
        pred_sh=anchor[te]+scale*sh*predict[te]
        scores[name]=metrics(pred_sh,test_truth,test_rw)
        scores[name+'_unshrunk']=metrics(pred_unsh,test_truth,test_rw)
        private[name]=pred_sh;private[name+'_unshrunk']=pred_unsh
        rec['families'][name]={'selection':best_config,'validation_trials':trials,'scores_shrunk':scores[name], 'scores_unshrunk':scores[name+'_unshrunk']}
    private_dir=out/'private';private_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(private_dir/(str(year)+'.npz'),**private)
    dump(out/'cells'/(str(year)+'.json'),rec)
    return rec,scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--out',required=True);ap.add_argument('--preflight',action='store_true'); args=ap.parse_args()
    d=open_verified(args.archive);feats,o,a,y,dates=build_features(d)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    code_sha=sha(Path(__file__).read_bytes())
    src={'schema':'EEX T3PA auction clearing price and completed auction published competition','sha256':SOURCE_SHA,'rows':len(d),
         'date_min':str(d.date.iloc[0].date()),'date_max':str(d.date.iloc[-1].date()),'target':'next auction clearing price EUR/tCO2',
         'known_at_origin':['current auction price','current auction cover/bid volumes','current bidder counts and auction dates'],
         'unknown_at_origin':['future auction cover','future realized volume','future bidders','future future price'],
         'policy':'strictly historical completed auction event values'}
    proto={'stage':'j','candidate':'signed_coverage_interaction_ridge','source':src,'source_script_sha256':code_sha,
           'test_years':YEARS,'lookback':LOOKBACK,'validation_events':VALIDATION_EVENTS,'alphas':ALPHAS,'shrinks':SHRINK,'models':MODELS,
           'information_setting':'after current completed auction t, predicting next completed auction t+1','data_status':'all six older years previously exposed as development, not untouched confirmation',
           'main_pair':'auction_additive vs signed_coverage_interaction_ridge, both same visible input/same search budget',
           'legacy':'separate task from old EUA_LONG single-step and national CEA daily price'}
    dump(out/'PROTOCOL.json',proto)
    if args.preflight:
        items=[]
        for year in YEARS:
            b=int(np.searchsorted(dates,np.datetime64(f'{year}-01-01')))
            e=int(np.searchsorted(dates,np.datetime64(f'{year+1}-01-01')))
            origin_test=o[(o>=b)&(o+1<e)];items.append({'year':year,'n_test_origins':len(origin_test),'first_origin':str(dates[origin_test[0]]),'first_label':str(dates[origin_test[0]+1]),'last_label':str(dates[origin_test[-1]+1])})
        dump(out/'PREFLIGHT.json',items);print('PREFLIGHT_OK',items);return
    allresults=[]
    for year in YEARS:
        st=time.monotonic();rec,sc=run_cell(d,feats,o,a,y,dates,year,out)
        allresults.append({'year':year,'n_test':rec['n_test'],'Q':{k:v['Q_MSE_RW'] for k,v in sc.items()},'RMSE':{k:v['RMSE'] for k,v in sc.items()},'duration_seconds':time.monotonic()-st})
        print('DONE',year,'n',rec['n_test'],'Q: ',{k:round(v['Q_MSE_RW'],4) for k,v in sc.items() if not k.endswith('_unshrunk')},flush=True)
    if sha(Path(__file__).read_bytes())!=code_sha:raise RuntimeError('Training source changed during execution')
    dump(out/'COMPLETE.json',{'status':'COMPLETE','completed_years':YEARS,'results':allresults,'source_sha256_unchanged':True,'source_script_sha256':code_sha})
    ranking=sorted([(k,float(np.mean([r['Q'][k] for r in allresults]))) for k in allresults[0]['Q']],key=lambda p:p[1]);print('MEAN_Q_RANKING',ranking)
if __name__=='__main__':main()
