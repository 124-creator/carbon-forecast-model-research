import json
from pathlib import Path
import numpy as np
root=Path(__file__).parent
rows=json.loads((root/'results.json').read_text())
if len(rows)!=36: raise AssertionError('expected 36 retained rows')
for row in rows:
    vals=[]
    if row['objective_calls']>180: raise AssertionError('evaluation cap exceeded')
    for d in row['details']:
        pred=np.array(d['predictions']); y=np.array(d['actual']); vals.append(np.mean((pred-y)**2))
        if d['support_max_target']>=d['query_min_origin']: raise AssertionError('immature support')
    if not np.isclose(np.mean(vals),row['test_loss'],rtol=1e-12): raise AssertionError('saved metrics mismatch')
print('PASS: 36 results; independent prediction-to-metric recomputation; mature supports; objective caps')
for row in rows:
    if row['method']=='candidate': print(row['process'],row['seed'],row['message'],[(d['collapse'],d['spectral_explosion'],d['spectral_radius']) for d in row['details']])
