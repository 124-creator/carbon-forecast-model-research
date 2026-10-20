import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).parent
rows=json.loads((OUT/'RESULTS.json').read_text());errors=[]
if len(rows)!=4:raise ValueError('four runs required')
for r in rows:
 z=np.load(OUT/f"{r['regime']}_{r['seed']}.npz",allow_pickle=False)
 score=float(np.mean((z['prediction'][256:]-z['target'][256:])**2));errors.append(abs(score-r['mse']))
 if abs(score-r['mse'])>1e-12:raise ValueError('score mismatch')
 limit=.5 if r['regime']=='nonlinear' else 1.1
 if not score<=limit*float(np.mean(z['target'][256:]**2)):raise ValueError('gate failed')
check=json.loads((OUT/'MAPPING.json').read_text())
if check['forward_max_error']>1e-10 or check['gradient_max_error']>2e-7:raise ValueError('mapping')
result=dict(runs=len(rows),max_score_error=max(errors),all_gates=True,optimized=not __debug__,scope='saved-score replay, not independent retraining')
(OUT/('VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json')).write_text(json.dumps(result,indent=2));print(result)
