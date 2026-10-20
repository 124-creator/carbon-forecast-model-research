import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).parent
errors=[];count=0
for r in json.loads((OUT/'RESULTS.json').read_text()):
 z=np.load(OUT/f"{r['regime']}_{r['seed']}_{r['model']}.npz",allow_pickle=False)
 score=float(np.mean((z['prediction'][256:]-z['target'][256:])**2));errors.append(abs(score-r['mse']));count+=1
for r in json.loads((OUT/'MARKET_RESULTS.json').read_text()):
 z=np.load(OUT/f"market_{r['seed']}_{r['model']}.npz",allow_pickle=False)
 if len(z['origins'])!=451:raise ValueError('origin count')
 vals=[]
 for year in ['2020','2021']:
  ok=z['years']==year;vals.extend((np.mean((z['prediction'][ok]-z['target'][ok])**2,0)/np.mean(z['target'][ok]**2,0)).tolist())
 errors.append(abs(float(np.mean(vals))-r['score']));count+=1
if count!=18 or max(errors)>1e-12:raise ValueError('saved-score mismatch')
result=dict(files=count,max_score_error=max(errors),optimized=not __debug__,scope='independent saved score replay, not independent retraining')
(OUT/('VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json')).write_text(json.dumps(result,indent=2));print(result)
