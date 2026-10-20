from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());logs=json.loads((R/'FIT_LOGS.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(z['origins'])!=451 or len(logs)!=24 or len({(v['seed'],v['block']) for v in logs})!=24:raise ValueError('count')
for v in logs:
    if v['max_validation_target']>=459+40*v['block'] or v['max_training_target']>=v['validation_first']:raise ValueError('maturity')
cells=[]
for yr in ['2020','2021']:
    mask=z['years']==yr
    for h in [1,5,20]:cells.append(np.mean([np.mean((z[f'lms_{s}'][mask,h-1]-z['truth'][mask,h-1])**2) for s in [17,29]])/np.mean(z['truth'][mask,h-1]**2))
if not np.isfinite(cells).all() or abs(np.mean(cells)-r['score'])>1e-12:raise ValueError('score')
if len(r['causal'])!=1 or not r['causal'][0]['passed']:raise ValueError('causal')
print(json.dumps(dict(verified=True,fits=24,origins=451,score=float(np.mean(cells)),parameters=logs[0]['parameters'],step_zero_fits=sum(v['selected']==0 for v in logs))))
