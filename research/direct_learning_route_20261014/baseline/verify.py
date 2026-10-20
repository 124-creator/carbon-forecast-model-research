from pathlib import Path
import json,sys
import numpy as np
R=Path(__file__).resolve().parent
z=np.load(R/'local_only/MARKET.npz',allow_pickle=False)
r=json.loads((R/'RUN_RECEIPT.json').read_text())
if len(z['origins'])!=451 or len(np.unique(z['origins']))!=451:raise RuntimeError('origin count')
if not np.all(z['dates']==np.sort(z['dates'])):raise RuntimeError('chronology')
if not all(d[:4] in ['2020','2021'] for d in z['dates']):raise RuntimeError('forbidden year')
for row in r['blocks']:
    if row['latest_label']>=row['fit_end'] or row['train_last']+20!=row['latest_label']:raise RuntimeError('maturity')
    if row['train_n']>480:raise RuntimeError('window')
    if row['model']=='HGBT' and any(c['min_support']<40 for c in row['capacity']):raise RuntimeError('HGBT support')
for a in r['perturbation']:
    if not a['actual_refit'] or not a['signature_equal'] or not a['prediction_equal']:raise RuntimeError('fit replay')
results={}
for kind in ['MBTR','HGBT']:
    pred=z[kind];ok=np.isfinite(pred).all(1)
    if int(ok.sum())!=r['completed'][kind]:raise RuntimeError('receipt completed mismatch')
    err=(pred-z['truth_price'])**2;rwerr=(z['rw_price']-z['truth_price'])**2
    translated=(pred-z['issuance_price'][:,None]-(z['truth_price']-z['issuance_price'][:,None]))**2
    np.testing.assert_allclose(err[ok],translated[ok],rtol=1e-12,atol=1e-10)
    cells=[];secondary=[]
    for year in ['2020','2021']:
        mask=(z['years']==year)&ok
        if not mask.any():continue
        mse=err[mask].mean(0);base=rwerr[mask].mean(0);ratio=mse/base
        secondary.extend(ratio.tolist())
        cells.extend([dict(year=year,h=h,n=int(mask.sum()),mse=float(mse[h-1]),rw_mse=float(base[h-1]),relative_mse=float(ratio[h-1])) for h in [1,5,20]])
    results[kind]=dict(completed=int(ok.sum()),complete_score=int(ok.sum())==451,cells=cells,primary_equal_year_h1_5_20=float(np.mean([c['relative_mse'] for c in cells])),secondary_equal_year_all20=float(np.mean(secondary)))
result=dict(passed=True,mode='normal' if __debug__ else 'optimized',results=results,fit_seconds=r['fit_seconds'],status=r['status'],historical_references_not_retrained=dict(MLE=.94979323,state_linear=.94270547),historical_saved_npz_access='denied; no retry; references from report only')
(R/('SCORES_NORMAL.json' if __debug__ else 'SCORES_O.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
