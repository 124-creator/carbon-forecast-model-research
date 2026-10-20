"""Independent saved-array scoring; no raw market file access."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
z=np.load(P/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False)
r=json.loads((P/'RESULTS.json').read_text());checks=[]
for row in r['results']:
 pred=z[row['name']];truth=z['truth'];cells=[]
 for year in sorted(set(z['years'])):
  m=z['years']==year;cells.extend((np.mean((pred[m]-truth[m])**2,axis=0)/np.mean(truth[m]**2,axis=0)).tolist())
 value=float(np.mean(cells));error=abs(value-row['score'])
 if error>1e-12:raise ValueError('score mismatch')
 checks.append(dict(name=row['name'],score=value,error=error))
controls=json.loads((P/'SYNTHETIC.json').read_text())
for row in controls:
 key=f"synthetic_{row['task']}_{row['seed']}";truth=z[key+'_truth']
 for kind,reported in row['ratio_rw'].items():
  value=float(np.mean((z[key+'_'+kind]-truth)**2)/np.mean(truth**2))
  if abs(value-reported)>1e-12:raise ValueError('control mismatch')
logs=json.loads((P/'TRAINING_LOGS.json').read_text());causal=json.loads((P/'CAUSALITY.json').read_text())
if len(z['origins'])!=451 or max(v['max_difference'] for v in causal)!=0:raise ValueError('causality/origins')
result=dict(passed=True,scoring_checks=checks,synthetic_rows=len(controls),max_normal_equation_gradient=max(v['gradient_norm'] for v in logs),max_future_perturbation=max(v['max_difference'] for v in causal),fit_count=len(logs),normal_or_optimized='optimized' if not __debug__ else 'normal')
(P/('VERIFY_OPTIMIZED.json' if not __debug__ else 'VERIFY_NORMAL.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
