from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());logs=json.loads((R/'FIT_LOGS.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(logs)!=48 or len({(v['length'],v['seed'],v['block']) for v in logs})!=48 or len(z['origins'])!=451:raise ValueError('count')
for v in logs:
    if v['max_validation_target']>=459+40*v['block'] or v['max_training_target']>=v['validation_first']:raise ValueError('maturity')
for row in r['scores']:
    vals=[]
    for yr in ['2020','2021']:
        mask=z['years']==yr
        for h in [1,5,20]:vals.append(np.mean([np.mean((z[f"L{row['length']}_{s}"][mask,h-1]-z['truth'][mask,h-1])**2) for s in [17,29]])/np.mean(z['truth'][mask,h-1]**2))
    if not np.isfinite(vals).all() or abs(np.mean(vals)-row['score'])>1e-12:raise ValueError('score')
if len(r['causal'])!=2 or not all(v['passed'] for v in r['causal']):raise ValueError('causal')
print(json.dumps(dict(verified=True,new_fits=48,origins=451,scores={str(v['length']):v['score'] for v in r['scores']})))
