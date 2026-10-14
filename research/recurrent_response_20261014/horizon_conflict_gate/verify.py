from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());logs=json.loads((R/'FIT_LOGS.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(logs)!=24 or len({(v['seed'],v['block']) for v in logs})!=24 or len(z['origins'])!=451:raise ValueError('count')
steps=0;negative=0
for v in logs:
    if v['max_validation_target']>=459+40*v['block'] or v['max_training_target']>=v['validation_first']:raise ValueError('maturity')
    steps+=len(v['history'])
    for a in v['history']:
        negative+=sum(c<0 for c in a['shared_cos'])
        if not np.isclose(a['raw_norm'],a['used_norm'],rtol=1e-5,atol=1e-6):raise ValueError('norm match')
if steps!=r['steps'] or 3*steps!=r['autograd_calls'] or negative!=r['shared_negative_pairs']:raise ValueError('budget')
for row in r['scores']:
    vals=[]
    for yr in ['2020','2021']:
        mask=z['years']==yr
        for h in [1,5,20]:vals.append(np.mean([np.mean((z[f"{row['kind']}_{s}"][mask,h-1]-z['truth'][mask,h-1])**2) for s in [17,29]])/np.mean(z['truth'][mask,h-1]**2))
    if not np.isfinite(vals).all() or abs(np.mean(vals)-row['score'])>1e-12:raise ValueError('score')
if len(r['causal'])!=1 or not r['causal'][0]['passed'] or not r['fixtures_passed']:raise ValueError('causal/fixtures')
print(json.dumps(dict(verified=True,fits=24,origins=451,negative_shared_pairs=negative,pair_comparisons=3*steps,negative_fraction=negative/(3*steps),scores={v['kind']:v['score'] for v in r['scores']})))
