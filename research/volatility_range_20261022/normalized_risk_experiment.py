"""EWMA97-scale normalized risk candidates and information-matched unconstrained control.
Reuse saved prior forecasts; never rerun old models or overwrite prior artifacts.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import corrected_risk_experiment as b
import gated_risk_experiment as g
ROOT=Path(__file__).resolve().parent
QMAP={'NQHAR_return':'QHAR_return','NQHAR_joint':'QHAR_joint','NQHAR_liquidity':'QHAR_liquidity','NQHAR_interaction':'QHAR_interaction'}
GATE=['N_gate_dynamic','N_gate_static']
CONTROL='QHAR_liquidity_plus_scale'
NAMES=list(QMAP)+GATE+[CONTROL]
CORRESP=dict(QMAP, N_gate_dynamic='QHAR_gate_dynamic',N_gate_static='QHAR_gate_static',QHAR_liquidity_plus_scale='QHAR_liquidity')
FLAGS=[];RFITS=[];RPARAM=[]

def flag(**record):
    FLAGS.append(record);print('FLAG '+json.dumps(record,default=str),flush=True)
    with (ROOT/'normalized_optimization_flags.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,default=str)+'\n')
b.flag=flag;g.flag=flag

def normalized_features(xs,scale):
    # Original three interaction columns remain identical: subtracting log(s)
    # from both own/range cancels in (range-own).
    result={}
    logscale=np.log(scale,where=scale>0,out=np.full_like(scale,np.nan))
    for name,orig in QMAP.items():
        xx=xs[orig].copy();count=3 if orig=='QHAR_return' else 7
        xx[:,:count]-=logscale[:,None]
        result[name]=xx
    result[CONTROL]=np.column_stack([xs['QHAR_liquidity'],logscale])
    return result

def run_asset(asset,prior):
    d,r,r2,eps,xs,ma,ew=b.feature_data(asset)
    n=len(d);scale=ew[.97]
    finite=np.logical_and.reduce([np.isfinite(v).all(axis=1) for v in xs.values()])
    assert np.all(np.isfinite(scale[finite])) and (scale[finite]>0).all()
    xx=normalized_features(xs,scale)
    years=d.date.dt.year.to_numpy();origins=np.flatnonzero((d.date>='2024-01-01')&(d.date<'2027-01-01'))
    targets={};eligible={}
    for h in b.HS:
        y=np.full(n,np.nan);same=np.zeros(n,dtype=bool)
        for t in range(n-h):y[t]=np.mean(r2[t+1:t+h+1]);same[t]=years[t]==years[t+h]
        targets[h]=y;eligible[h]=same
    fits={};rows=[];weights=[]
    for j,t in enumerate(origins):
        if j%22==0:
            for h in b.HS:
                train=np.flatnonzero(finite&eligible[h]&np.isfinite(targets[h])&(np.arange(n)+h<=t))
                assert np.max(train+h)<=t
                ctx={'asset':asset,'origin':str(d.date.iloc[t].date()),'origin_index':int(t),'horizon':h,'train_origin_max':int(train[-1]),'train_label_end_max':int(train[-1]+h)}
                for name in list(QMAP)+[CONTROL]:
                    ytrain=targets[h][train] if name==CONTROL else targets[h][train]/scale[train]
                    fits[h,name],meta=b.fit_regression(name,xx[name][train],ytrain,eps,ctx)
                    RFITS.append(dict(ctx,model=name,n_train=len(train),**meta))
                    mu,sd,beta,smear=fits[h,name]
                    RPARAM.append(dict(ctx,model=name,mu=json.dumps(mu.tolist()),sd=json.dumps(sd.tolist()),parameters=json.dumps(beta.tolist())))
                for name in GATE:
                    fits[h,name]=g.fit_gate(xx['NQHAR_liquidity'][train],targets[h][train]/scale[train],name.endswith('dynamic'),dict(ctx,model=name))
            print(f'NORMALIZED {asset} refit {j//22+1} {d.date.iloc[t].date()}',flush=True)
        for h in b.HS:
            old=prior.loc[(asset,t,h)]
            assert np.isclose(old.target,targets[h][t],rtol=1e-12,atol=1e-16,equal_nan=True)
            assert bool(old.scored_common)==bool(eligible[h][t] and np.isfinite(targets[h][t]))
            for name in NAMES:
                if name in GATE:
                    scaler,theta=fits[h,name];own,rng,direct=g.transform(xx['NQHAR_liquidity'][t:t+1],scaler)
                    eta,w,_,_=g.components(theta,own,rng,direct,name.endswith('dynamic'));raw=float(np.exp(eta[0]))
                    weights.append({'asset':asset,'date':old.date,'origin_index':int(t),'year':int(years[t]),'horizon':h,'model':name,'gate_weight':float(w[0]),'scored_common':bool(old.scored_common)})
                else:
                    raw=b.regression_prediction(name,fits[h,name],xx[name][t],eps,{'asset':asset,'origin':old.date,'horizon':h})
                pred=raw if name==CONTROL else scale[t]*raw
                assert np.isfinite(pred) and pred>0
                rec=old.to_dict();rec.update(asset=asset,origin_index=int(t),horizon=h,model=name,forecast=pred,risk_scale=float(scale[t]),normalized_forecast=raw if name!=CONTROL else np.nan)
                rec['qlike']=float(np.log(pred)+old.target/pred) if old.scored_common else np.nan
                rec['squared_error']=float((pred-old.target)**2) if old.scored_common else np.nan
                if old.scored_common and name!=CONTROL:
                    np.testing.assert_allclose(rec['qlike'],np.log(scale[t])+np.log(raw)+(old.target/scale[t])/raw,rtol=1e-12,atol=1e-12)
                rows.append(rec)
    return rows,weights

def main():
    start=time.time();(ROOT/'normalized_optimization_flags.jsonl').write_text('',encoding='utf-8')
    old=pd.read_csv(ROOT/'predictions_all.csv');gateold=pd.read_csv(ROOT/'gate_predictions_all.csv')
    prior=old[old.model=='MA22'].set_index(['asset','origin_index','horizon']);assert prior.index.is_unique
    rows=[];weights=[]
    for asset in b.ASSETS:
        rr,ww=run_asset(asset,prior);rows.extend(rr);weights.extend(ww)
    new=pd.DataFrame(rows);new.to_csv(ROOT/'normalized_predictions_all.csv',index=False)
    allp=pd.concat([old,gateold,new],ignore_index=True);common=allp[allp.scored_common].copy()
    models=b.MODELS+g.NAMES+NAMES
    assert (common.groupby(['asset','date','horizon']).model.nunique()==len(models)).all()
    assert not allp.duplicated(['asset','date','horizon','model']).any()
    common.to_csv(ROOT/'normalized_comparison_predictions_common.csv',index=False)
    summary=[];pair=[]
    for (asset,year,stage,h),group in common.groupby(['asset','year','stage','horizon']):
        loss=group.pivot(index='date',columns='model',values='qlike');mse=group.pivot(index='date',columns='model',values='squared_error')
        assert loss.notna().all().all()
        for model in models:
            dd=loss[model]-loss.MA22
            summary.append({'asset':asset,'year':year,'stage':stage,'horizon':h,'model':model,'n':len(loss),'qlike':loss[model].mean(),'delta_MA22':dd.mean(),'NW_t_vs_MA22':b.nw_t(dd,max(5,h-1)),'MSE':mse[model].mean(),'MSE_pct_MA22':100*mse[model].mean()/mse.MA22.mean()})
        for model in NAMES:
            for benchmark in models:
                if benchmark==model:continue
                dd=loss[model]-loss[benchmark]
                pair.append({'asset':asset,'year':year,'stage':stage,'horizon':h,'model':model,'benchmark':benchmark,'n':len(loss),'delta_QLIKE':dd.mean(),'NW_t':b.nw_t(dd,max(5,h-1)),'NW_lag':max(5,h-1),'MSE_ratio_pct':100*mse[model].mean()/mse[benchmark].mean()})
    summary=pd.DataFrame(summary);pair=pd.DataFrame(pair)
    summary.to_csv(ROOT/'normalized_summary.csv',index=False);pair.to_csv(ROOT/'normalized_pairwise.csv',index=False)
    aggregate=pair.groupby(['year','horizon','model','benchmark'],as_index=False).agg(delta_QLIKE_equal_asset=('delta_QLIKE','mean'),MSE_ratio_pct_equal_asset=('MSE_ratio_pct','mean'))
    aggregate.to_csv(ROOT/'normalized_pairwise_equal_asset.csv',index=False)
    pd.DataFrame(RFITS).to_csv(ROOT/'normalized_regression_fit_diagnostics.csv',index=False)
    pd.DataFrame(RPARAM).to_csv(ROOT/'normalized_regression_parameters.csv',index=False)
    pd.DataFrame(g.DIAGS).to_csv(ROOT/'normalized_gate_fit_diagnostics.csv',index=False)
    pd.DataFrame(g.PARAMS).to_csv(ROOT/'normalized_gate_parameters.csv',index=False)
    weight=pd.DataFrame(weights);weight.to_csv(ROOT/'normalized_gate_weights.csv',index=False)
    qs=weight[weight.scored_common].groupby(['asset','year','horizon','model']).gate_weight.agg(n='size',mean='mean',minimum='min',q05=lambda ss:ss.quantile(.05),median='median',q95=lambda ss:ss.quantile(.95),maximum='max').reset_index()
    qs.to_csv(ROOT/'normalized_gate_weight_summary.csv',index=False)
    outputs=['normalized_predictions_all.csv','normalized_comparison_predictions_common.csv','normalized_summary.csv','normalized_pairwise.csv','normalized_pairwise_equal_asset.csv','normalized_regression_fit_diagnostics.csv','normalized_regression_parameters.csv','normalized_gate_fit_diagnostics.csv','normalized_gate_parameters.csv','normalized_gate_weights.csv','normalized_gate_weight_summary.csv','normalized_optimization_flags.jsonl']
    manifest={'runtime_seconds':time.time()-start,'script_sha256':b.digest(Path(__file__)),'base_script_sha256':b.digest(ROOT/'corrected_risk_experiment.py'),'gate_script_sha256':b.digest(ROOT/'gated_risk_experiment.py'),'old_predictions_sha256':b.digest(ROOT/'predictions_all.csv'),'old_gate_predictions_sha256':b.digest(ROOT/'gate_predictions_all.csv'),'outputs':{f:b.digest(ROOT/f) for f in outputs},
      'models':NAMES,'corresponding_original_models':CORRESP,
      'normalization':'s_s=causal EWMA97 variance at each historical origin s; train response Y_s/s_s; subtract log(s_s) from first3 own log-features and first7 own/range/gap log-features, respectively; rel,flat,range-minus-own interactions unchanged; final forecast s_t*exp(g_t). Original stored targets and scoring remain unmodified.',
      'control':'QHAR_liquidity_plus_scale has original nine liquidity features plus log(s_t), trains on original Y, no fixed offset. Separates added EWMA information from homogeneity/coefficient constraints, but standardization/ridge geometry also differs.',
      'constraint':'Normalized linear QHAR expands to a log-scale offset with coefficient 1-minus-sum of proxy slopes on log(s). It is not identical to unnormalized QHAR; unchanged QLIKE loss does not imply identical hypothesis class.',
      'training':'Same original expanding 22-origin refit, s+h<=refit_t, cross-year training/scoring exclusion, epsilon, standardization and ridge convention; QLIKE mean loss+.01/2 slope norm squared; fixed hyperparameters; gate initial .2/.5/.8 selected on training only',
      'checks':{'original_targets_unchanged':True,'all18_models_identical_scoring_origins':True,'normalized_QLIKE_equals_original_minus_logscale':True,'scale_strictly_positive_no_floor_no_target_fill':True},
      'counts':{'new_prediction_rows':len(new),'all_scored_rows':len(common),'common_asset_horizon_date_origins':len(common)//len(models),'regression_fits':len(RFITS),'gate_start_fits':len(g.DIAGS),'selected_gate_fits':len(g.PARAMS),'flags':len(FLAGS)},
      'limits':['All years seen; no untouched test','No novelty claim; prior art remains to be checked','Gate nonconvex with deterministic local multistarts','NW t exploratory with overlap and multiple comparisons; no multiplicity adjustment','Scaling changes constraints and penalty geometry, not only numeric units','Original raw data and coverage limitations persist']}
    (ROOT/'normalized_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('NORMALIZED COMPLETE '+json.dumps(manifest['counts']),flush=True)
    focus=aggregate[(aggregate.model.isin(['NQHAR_liquidity','N_gate_dynamic']))&aggregate.benchmark.isin(['EWMA97','QHAR_liquidity','QHAR_liquidity_plus_scale','QHAR_gate_dynamic','N_gate_static'])]
    print(focus.to_string(index=False),flush=True)

if __name__=='__main__':main()
