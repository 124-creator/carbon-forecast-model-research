import json,sys,hashlib
from pathlib import Path
import numpy as np
import run as m
OUT=Path(__file__).resolve().parent

def main():
 dates,p=m.base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 z=np.load(OUT/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
 if len(orig)!=451 or not np.array_equal(z['origins'],orig):raise ValueError('origin protocol')
 y=p[orig[:,None]+m.H]-p[orig,None]
 np.testing.assert_array_equal(z['true_changes'],y)
 hist,state=m.history(p,initial);np.testing.assert_allclose(z['issuance_mle'],hist);np.testing.assert_allclose(z['states'],state)
 results=json.loads((OUT/'ALL_RESULTS.json').read_text(encoding='utf-8'));cs=m.configurations()
 if len(results)!=len(cs):raise ValueError('configuration omission')
 cells=[];replay_count=0
 for r,c in zip(results,cs):
  if c!=r['configuration']:raise ValueError('configuration mismatch')
  pred=z[c['name']];values=[]
  for year in ['2020','2021']:
   mask=np.char.startswith(dates[orig],year);rel=np.mean((pred[mask]-y[mask])**2,0)/np.mean(y[mask]**2,0);values.extend(rel.tolist())
   target=next(row for row in r['by_year'] if row['year']==year)
   np.testing.assert_allclose(rel,target['relative_mse_h'],rtol=1e-12,atol=1e-12)
  np.testing.assert_allclose(np.mean(values),r['score'],rtol=1e-12,atol=1e-12)
  replay=hist[orig].copy()
  if c['family']!='MLE':
   for fit in range(initial,len(p),40):
    pos=np.where((orig>=fit)&(orig<fit+40))[0]
    if len(pos):replay[pos],_=m.correct(p,hist,state,fit,orig[pos],c)
  np.testing.assert_allclose(pred,replay,rtol=1e-12,atol=1e-12);replay_count+=1
 logs=json.loads((OUT/'TRAINING_LOGS.json').read_text(encoding='utf-8'))
 for log in logs:
  if log['max_target']>=log['fit'] or log['max_origin']+log['h']!=log['max_target']:raise ValueError('maturity')
 # Actual MLE residual diagnostics: sign/state strata, not new selection.
 for year in ['2020','2021']:
  for k,h in enumerate(m.H):
   for sign in [-1,1]:
    for gsign in [-1,1]:
     mask=np.char.startswith(dates[orig],year)&(np.sign(hist[orig,k])==sign)&(np.sign(state[orig,0])==gsign)
     if np.any(mask):cells.append(dict(year=year,h=int(h),mle_sign=sign,state_sign=gsign,n=int(mask.sum()),mean_error=float(np.mean(hist[orig,k][mask]-y[:,k][mask])),mse=float(np.mean((hist[orig,k][mask]-y[:,k][mask])**2))))
 (OUT/'MLE_DIAGNOSTICS.json').write_text(json.dumps(cells,indent=2),encoding='utf-8')
 result=dict(passed=True,origins=451,configurations=len(cs),replayed_configurations=replay_count,maturity_requests=len(logs),mode='optimized' if not __debug__ else 'normal',independent_original_price_rescore=True)
 filename='VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json'
 (OUT/filename).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
