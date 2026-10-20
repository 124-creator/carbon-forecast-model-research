import json
import numpy as np
from pathlib import Path
from market_parent import score
O=Path(__file__).parent
z=np.load(O/'MARKET_LOCAL_ONLY.npz',allow_pickle=False)
rows=json.loads((O/'MARKET_RESULTS.json').read_text());logs=json.loads((O/'MARKET_LOGS.json').read_text())
if len(rows)!=5 or len(z['origins'])!=451:raise ValueError('count')
for r in rows:
 e=abs(score(z[f"{r['model']}_{r['seed']}"],z['truth'],z['years'])-r['score'])
 if e>1e-12:raise ValueError('metric')
for l in logs:
 if l['train_max_target']>=l['fit'] or l['validation_max_target']>=l['fit']:raise ValueError('maturity')
print({'scores_verified':len(rows),'fits':len(logs),'scope':'saved-score and boundary checks, no independent retraining or perturbation test'})
