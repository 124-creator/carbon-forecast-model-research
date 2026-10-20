import json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).parent
z=np.load(P/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
def req(c,m):
 if not c:raise ValueError(m)
results=json.loads((P/'RESULTS.json').read_text(encoding='utf8'));logs=json.loads((P/'TRAINING_LOGS.json').read_text(encoding='utf8'));checks=json.loads((P/'CAUSAL_REFITS.json').read_text(encoding='utf8'))
req(len(z['origins'])==451,'origins');req(z['dates'][-1]<='2021-12-31','price date')
maxdiff=0
for r in results:
 key='MLE' if r['name']=='MLE480' else r['name'];pr=z[key];vals=[]
 req(np.isfinite(pr).all(),'finite')
 for y in ['2020','2021']:
  mask=z['years']==y;rel=np.mean((pr[mask]-z['truth'][mask])**2,axis=0)/np.mean(z['truth'][mask]**2,axis=0);vals.extend(rel)
 diff=abs(float(np.mean(vals))-r['score']);maxdiff=max(maxdiff,diff);req(diff<1e-12,'score')
 if key!='MLE':
  lag=key[-1];valid=z['valid_lag'+lag][z['origins']];req(np.array_equal(pr[~valid],z['MLE'][~valid]),'fallback')
for l in logs:req(l['inner_last_target']<l['validation_first'] and l['outer_last_target']<l['end'],'maturity')
for c in checks:req(c['param_diff']<1e-12 and c['pred_diff']<1e-12,'refit')
req({c['end'] for c in checks if c['kind']=='future_price'}=={459,499,659},'representative refits')
receipt=dict(passed=True,optimized=not __debug__,scores=len(results),origins=451,max_score_difference=maxdiff,refit_checks=len(checks))
(P/('VERIFIED_OPT.json' if not __debug__ else 'VERIFIED_NORMAL.json')).write_text(json.dumps(receipt,indent=2),encoding='utf8');print(receipt)
