"""Parent-callable independent scoring of audit predictions; no source prices read."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
z=np.load(OUT/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
rows=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'));checks=json.loads((OUT/'CHECKS.json').read_text(encoding='utf-8'))
if z['truth'].shape!=(451,3):raise ValueError('shape')
errors={}
for r in rows:
 ratios=[]
 for y in sorted(set(z['years'])):
  mask=z['years']==y;ratios.extend((np.mean((z[r['name']][mask]-z['truth'][mask])**2,axis=0)/np.mean(z['truth'][mask]**2,axis=0)).tolist())
 errors[r['name']]=abs(float(np.mean(ratios))-r['score'])
if max(errors.values())>1e-12:raise ValueError(errors)
if abs(rows[0]['score']-.9427054678157569)>1e-12:raise ValueError('historical')
if abs(rows[-1]['score']-1.2498356686412873)>1e-12:raise ValueError('local')
if not checks['state_origins_equal'] or not checks['local_origins_equal']:raise ValueError('origins')
if any(v>1e-12 for k,v in checks.items() if k.endswith('maxdiff')):raise ValueError('checks')
result=dict(passed=True,independent_score_errors=errors,historical=rows[0]['score'],local=rows[-1]['score'],original_saved_predictions_reproduced=True,source_data_read=False)
(OUT/('PARENT_VALIDATION_OPTIMIZED.json' if not __debug__ else 'PARENT_VALIDATION.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
