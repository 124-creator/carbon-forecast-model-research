import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
def req(b,s):
 if not b:raise ValueError(s)
z=np.load(OUT/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);logs=json.loads((OUT/'TRAINING_LOGS.json').read_text(encoding='utf8'));checks=json.loads((OUT/'FUTURE_REFITS.json').read_text(encoding='utf8'))
req(len(z['origins'])==451 and np.all(np.diff(z['origins'])>0),'origins');req(set(z['years'])=={'2020','2021'},'years');results=[]
for name in ['MLE','PRICE','COAL','FX']:
 req(z[name].shape==(451,3) and np.isfinite(z[name]).all(),'shape finite');rows=[]
 for y in ['2020','2021']:
  m=z['years']==y;den=np.mean(z['truth'][m]**2,axis=0);mse=np.mean((z[name][m]-z['truth'][m])**2,axis=0);rows.append(dict(year=y,n=int(m.sum()),mse_h=mse.tolist(),rw_mse_h=den.tolist(),relative_mse_h=(mse/den).tolist()))
 results.append(dict(name=name,score=float(np.mean([r['relative_mse_h'] for r in rows])),by_year=rows))
req(abs(results[0]['score']-.9497932301207621)<1e-12,'MLE')
for name,mask in [('COAL','valid_coal'),('FX','valid_fx')]:req(np.array_equal(z[name][~z[mask]],z['PRICE'][~z[mask]]),'missing fallback')
for l in logs:req(l['max_target']<l['end'] and l['inner_n']>20,'maturity')
req(len(checks)==9 and all(x['parameter_diff']<1e-12 and x['prediction_diff']<1e-12 and x['selection_same'] for x in checks),'9 actual refits')
counts={name:{str(a):sum((l['price_alpha'] if j==0 else l['source_alpha'][j-1])==a for l in logs) for a in [None,.1,10.]} for j,name in enumerate(['PRICE','COAL','FX'])}
summary=dict(results=results,selection_counts=counts,refits_verified=len(checks),mature_refreshes=len(logs),historical_strongstate=.942705,coal_increment=results[2]['score']-results[1]['score'],fx_increment=results[3]['score']-results[1]['score'])
(OUT/('VERIFICATION_NORMAL.json' if __debug__ else 'VERIFICATION_OPT.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
