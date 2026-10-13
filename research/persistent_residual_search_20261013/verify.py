import importlib.util,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('persistent',OUT/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def eq(a,b):
 if not np.allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True):raise ValueError('replay mismatch')
def main():
 dates,p=m.old.base.read_prefix();initial=int(np.sum(dates<'2020-01-01'));orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 z=np.load(OUT/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);eq(orig,z['origins']);hist,s=m.old.history(p,initial);eq(hist,z['issuance_mle'])
 y=p[orig[:,None]+m.H]-p[orig,None];eq(y,z['true_changes']);years=np.array([d[:4] for d in dates[orig]])
 results=json.loads((OUT/'ALL_RESULTS.json').read_text(encoding='utf-8'))
 if len(results)!=8:raise ValueError('omitted config')
 for r in results:
  pred,_=m.replay(p,hist,s,orig,initial,r['name']);eq(pred,z[r['name']]);sc,rows=m.old.base.score(pred,y,years);eq(sc,r['score'])
  for a,b in zip(rows,r['by_year']):eq(a['relative_mse_h'],b['relative_mse_h'])
 labels=np.full(hist.shape,np.nan)
 for k,h in enumerate(m.H):labels[:-h,k]=p[h:]-p[:-h]
 perturb=0
 for fit in range(initial,len(p),40):
  for name in m.NAMES[2:]:
   b=m.bias(labels,hist,fit,name)
   for k,h in enumerate(m.H):
    bad=labels.copy();bad[fit-int(h):,k]=1234567;eq(b,m.bias(bad,hist,fit,name));perturb+=1
 for fit in [initial,initial+80]:
  q=p.copy();q[fit:]+=1000;mh,sh=m.old.history(q,initial);eq(mh[:fit],hist[:fit]);eq(sh[:fit],s[:fit])
  for name in m.NAMES[2:]:
   ly=labels.copy()
   for k,h in enumerate(m.H):ly[:-h,k]=q[h:]-q[:-h]
   eq(m.bias(labels,hist,fit,name),m.bias(ly,mh,fit,name))
 rejected=0
 for bad,fit in [(labels,100),(labels[:10],initial),(np.full(labels.shape,np.nan),initial)]:
  try:m.bias(bad,hist,fit,'age_shared')
  except ValueError:rejected+=1
 if rejected!=3:raise ValueError('exception validation failed')
 result=dict(passed=True,configurations=8,original_price_rescore=True,replay=True,unmatured_label_perturbations=perturb,explicit_invalid_cases=rejected,future_price_tests=True,mode='normal' if __debug__ else 'optimized')
 m.dump('VERIFY_NORMAL.json' if __debug__ else 'VERIFY_OPTIMIZED.json',result);print(json.dumps(result))
if __name__=='__main__':main()
