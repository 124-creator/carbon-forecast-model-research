"""Independent read-only output checks; writes only a validation receipt."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import gdea_risk_experiment as g
root=Path(__file__).resolve().parent
rng=np.random.default_rng(421)
random_checks=0
for k in [2,5,7]:
    for h in [1,5,22]:
        rr=rng.lognormal(-8,1,100); origins=np.arange(1,61); t=90
        x=rng.normal(-8,1,(60,k)); decay=np.exp2(-(t-origins)/22)
        y=np.array([rr[s+1:s+h+1].mean() for s in origins]); by,a=g.lb(x,origins,t,h,rr,decay)
        for j in range(4):
            w=rng.dirichlet(np.ones(k)); fv,fg=g.objective(w,x,y,decay/decay.sum()); bv,bg=g.objective(w,x,by,a/a.sum())
            np.testing.assert_allclose(fv,bv,rtol=1e-11,atol=1e-11); np.testing.assert_allclose(fg,bg,rtol=1e-11,atol=1e-11)
            random_checks+=1
p=pd.read_csv(root/'predictions_common.csv',float_precision='round_trip'); b=pd.read_csv(root/'base_predictions_all.csv',float_precision='round_trip'); w=pd.read_csv(root/'weights.csv',float_precision='round_trip')
d,r,r2,eps,xs,ma,ew=g.feature_data('GDEA'); years=d.date.dt.year.to_numpy()
maxdiff=0.
for h,sub in p.groupby('horizon'):
    for row in sub[sub.model=='MA22'].itertuples():
        t=row.origin_index; target=r2[t+1:t+h+1].mean(); maxdiff=max(maxdiff,abs(target-row.target))
        np.testing.assert_allclose(row.target,target,rtol=5e-13,atol=1e-18)
        np.testing.assert_allclose(row.forecast,r2[t-21:t+1].mean(),rtol=5e-13,atol=1e-18)
        assert years[t]==years[t+h] and str(d.date.iloc[t+h].date())==row.target_end
    assert sub.groupby('origin_index').model.nunique().eq(14).all()
for row in b.itertuples():
    assert row.refit_origin_index<=row.origin_index and row.origin_index-row.refit_origin_index<22
    assert row.scored_common==bool(row.origin_index+row.horizon<len(d) and years[row.origin_index]==years[min(row.origin_index+row.horizon,len(d)-1)])
for scope in ['dynamic','expanding']:
    z=w[w.horizon==1]
    a=z[z.model=='full_'+scope].sort_values('origin_index')[g.EXPERTS].to_numpy()
    for kind in ['partial','LB']:
        bb=z[z.model==kind+'_'+scope].sort_values('origin_index')[g.EXPERTS].to_numpy()
        np.testing.assert_allclose(a,bb,rtol=1e-8,atol=1e-8)
assert (w.observed_end_max<=w.origin_index).all()
# Reconstruct one partially observed window for every horizon/scope, validating
# actual coefficients directly against additive objective and gradient.
partial_checks=0
for h in [1,5,22]:
    ss=p[(p.horizon==h)&(p.model=='MA22')].sort_values('origin_index')
    origins=ss.origin_index.to_numpy(int); i=min(350,len(origins)-1); t=origins[i]
    forecasts=p[p.horizon==h].pivot(index='origin_index',columns='model',values='forecast').loc[origins,g.EXPERTS].to_numpy(); x=np.log(forecasts)
    for scope in ['dynamic','expanding']:
        pool=np.arange(i); pool=pool[-252:] if scope=='dynamic' else pool
        decay=np.exp2(-(t-origins[pool])/22) if scope=='dynamic' else np.ones(len(pool))
        m=np.minimum(h,t-origins[pool]); alpha=m/h
        known=np.array([r2[s+1:s+mm+1].sum() for s,mm in zip(origins[pool],m)])
        ww=rng.dirichlet(np.ones(g.K)); eta=x[pool]@ww; Z=decay@alpha
        direct=float(decay@(alpha*eta+known/h*np.exp(-eta))/Z+.005*np.sum((ww-g.U)**2))
        directgrad=x[pool].T@(decay*(alpha-known/h*np.exp(-eta))/Z)+.01*(ww-g.U)
        val,grad=g.objective(ww,x[pool],known/m,decay*alpha/Z)
        np.testing.assert_allclose(val,direct,rtol=1e-11,atol=1e-11); np.testing.assert_allclose(grad,directgrad,rtol=1e-11,atol=1e-11)
        partial_checks+=1
receipt={'random_all_mature_LB_objective_gradient_checks_K2_5_7':random_checks,'actual_partial_additive_checks':partial_checks,'all_targets_and_MA22_reconstructed':True,'max_target_csv_difference':maxdiff,'h1_all_scopes_weights_verified':True,'refit_cadence_and_scored_mask_verified':True,'no_future_observed_components':True,'source_sha256':g.digest(g.SOURCE),'prediction_sha256':g.digest(root/'predictions_common.csv'),'no_models_refitted':True}
g.save('independent_validation.json',receipt)
print(json.dumps(receipt,indent=2))
