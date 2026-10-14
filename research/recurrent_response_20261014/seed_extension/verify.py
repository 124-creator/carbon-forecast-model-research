import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False);y=z['truth'];years=z['years']
if len(y)!=451 or r['fits']!=24:raise ValueError('count')
logs=json.loads((R/'FIT_LOGS.json').read_text())
if len({(v['seed'],v['block']) for v in logs})!=24:raise ValueError('fits')
for v in logs:
    if v['max_validation_target']>=459+40*v['block'] or v['max_training_target']>=v['validation_first']:raise ValueError('purge')
for v in r['receipts']:
    if v['updates']!=470 or v['labels']!=9210:raise ValueError('online budget')
def score(a):
    return float(np.mean([np.mean((a[years==yr,h-1]-y[years==yr,h-1])**2)/np.mean(y[years==yr,h-1]**2) for yr in ['2020','2021'] for h in [1,5,20]]))
for k,v in r['scores'].items():
    if abs(score(z[k])-v)>1e-12 or not np.isfinite(z[k]).all():raise ValueError('score')
for kind,v in r['groups'].items():
    if abs(score(np.mean([z[f'{kind}_{s}'] for s in [17,29,41,53]],axis=0))-v['prediction_ensemble'])>1e-12:raise ValueError('ensemble')
if len(r['causal'])!=2 or not all(v['passed'] for v in r['causal']):raise ValueError('causal')
print(json.dumps(dict(verified=True,new_fits=24,new_online_rollouts=2,origins=451,extra_seeds={str(s):{k:r['scores'][f'{k}_{s}'] for k in ['slow','fast','average','delayed_weight']} for s in [41,53]},fit_seconds=sum(v['seconds'] for v in logs),online_seconds=sum(v['seconds'] for v in r['receipts'])),indent=2))
