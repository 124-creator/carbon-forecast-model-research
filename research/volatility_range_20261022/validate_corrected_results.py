from pathlib import Path
import json
import numpy as np
import pandas as pd
from arch import arch_model
import corrected_risk_experiment as b

root=Path(__file__).resolve().parent
p=pd.read_csv(root/'predictions_all.csv')
fits=pd.read_csv(root/'fit_diagnostics.csv')
summary=pd.read_csv(root/'summary.csv')
report={'checks':{},'flat_zero_volume':[],'garch_midblock_checks':[]}
assert len(p)==len(p.drop_duplicates(['asset','date','horizon','model']))
assert (p.groupby(['asset','date','horizon']).model.nunique()==9).all()
score=p[p.scored_common]
assert score.forecast.gt(0).all() and score.target.ge(0).all()
np.testing.assert_allclose(score.qlike,np.log(score.forecast)+score.target/score.forecast,rtol=1e-10,atol=1e-11)
np.testing.assert_allclose(score.squared_error,(score.forecast-score.target)**2,rtol=1e-9,atol=1e-15)
for asset in b.ASSETS:
    d=pd.read_csv(b.DATA/f'{asset}_ohlc.csv',parse_dates=['date'])
    r=np.log(d.adjclose.to_numpy()[1:]/d.adjclose.to_numpy()[:-1]);r2=r*r
    flat=d.high.eq(d.low)
    report['flat_zero_volume'].append({'asset':asset,'flat':int(flat.sum()),'flat_and_zero_volume':int((flat&d.volume.eq(0)).sum()),'pct_zero_volume_given_flat':float(100*(flat&d.volume.eq(0)).sum()/flat.sum())})
    a=p[(p.asset==asset)&(p.model=='MA22')]
    for rec in a.itertuples():
        t=rec.origin_index;h=rec.horizon
        expected=np.mean(r2[t:t+h]) if t+h<len(d) else np.nan
        np.testing.assert_allclose(rec.target,expected,rtol=1e-10,atol=1e-16,equal_nan=True)
        np.testing.assert_allclose(rec.forecast,np.mean(r2[t-22:t]),rtol=1e-10,atol=1e-16)
        correct=t+h<len(d) and d.date.iloc[t].year==d.date.iloc[t+h].year
        assert rec.scored_common==correct
    gf=fits[(fits.asset==asset)&(fits.model=='GARCH11_t')]
    # Independently ask arch to filter all observations at fixed saved parameters
    # midway through each refit block; compare with our persisted recurrence.
    for ref in gf.itertuples():
        t=min(ref.origin_index+11,len(d)-1)
        am=arch_model(r[:t]*100,mean='Zero',vol='GARCH',p=1,q=1,dist='StudentsT',rescale=False)
        fixed=am.fix(np.array([ref.omega,ref.alpha,ref.beta,ref.nu]))
        expected=fixed.forecast(horizon=22,method='analytic',reindex=False).variance.iloc[-1].to_numpy()/10000
        for h in [1,5,22]:
            got=p[(p.asset==asset)&(p.origin_index==t)&(p.horizon==h)&(p.model=='GARCH11_t')].forecast.iloc[0]
            np.testing.assert_allclose(got,np.mean(expected[:h]),rtol=1e-9,atol=1e-15)
    report['garch_midblock_checks'].append({'asset':asset,'blocks':len(gf),'horizons_per_block':3})
reg=fits[fits.model!='GARCH11_t']
assert (reg.train_label_end_max<=reg.origin_index).all()
assert (reg.train_origin_max+reg.horizon==reg.train_label_end_max).all()
for rec in summary.itertuples():
    g=score[(score.asset==rec.asset)&(score.year==rec.year)&(score.horizon==rec.horizon)&(score.model==rec.model)]
    assert len(g)==rec.n
    np.testing.assert_allclose(g.qlike.mean(),rec.qlike,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(g.squared_error.mean(),rec.MSE,rtol=1e-10,atol=1e-15)
report['checks']={'direct_raw_target_and_MA22':True,'all9_models_common':True,'no_cross_year':True,'saved_loss_recalculation':True,'saved_summary_recalculation':True,'training_label_boundaries':True,'garch_fixed_filter_midblock_vs_arch':True}
report['zero_target_origin_counts']=score[score.model=='MA22'].groupby(['asset','horizon']).target.apply(lambda s:int((s==0).sum())).to_dict().__repr__()
(root/'independent_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
sel=summary[(summary.year.isin([2025,2026]))&(summary.model.isin(['EWMA97','GARCH11_t','QHAR_joint','QHAR_liquidity','QHAR_interaction']))][['asset','year','horizon','model','n','delta_MA22','NW_t_vs_MA22','delta_liquidity','NW_t_vs_liquidity','MSE_pct_MA22']]
sel.to_csv(root/'focused_asset_results.csv',index=False)
print(json.dumps(report,indent=2,ensure_ascii=False))
print(sel.to_string(index=False))
