"""Independent saved-result verifier: never opens price CSV."""
import json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
a=np.load(p/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
r=json.loads((p/'RESULTS.json').read_text())
errors=[]
for v in r:
    ratios=[]
    for c in v['cells']:
        m=np.char.startswith(a['dates'],c['year']);k=[1,5,20].index(c['h'])
        mse=float(np.mean((a[v['name']][m,k]-a['target'][m,k])**2))
        rw=float(np.mean(a['target'][m,k]**2));ratios.append(mse/rw)
        errors.extend([abs(mse-c['original_price_mse']),abs(rw-c['rw_mse'])])
    errors.append(abs(float(np.mean(ratios))-v['score']))
if len(a['origins'])!=451 or max(errors)>1e-12:raise ValueError('score mismatch')
for n in [1,40]:
    if not np.array_equal(a[f'common_refresh{n}'][:,2],a[f'per_h_refresh{n}'][:,2]):raise ValueError('h20 mismatch')
print(json.dumps(dict(origins=len(a['origins']),configurations=len(r),max_error=max(errors),h20_support_equal=True),indent=2))
