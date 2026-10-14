from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(z['origins'])!=451 or len(r['receipts'])!=4:raise ValueError('count')
for v in r['receipts']:
    if v['updates']!=451 or v['label_contributions']!=9020:raise ValueError('budget')
    logs=json.loads((R/f"{v['kind']}_{v['seed']}_LOG.json").read_text())
    if len(logs)!=451 or [a['t'] for a in logs]!=list(range(479,930)):raise ValueError('logs')
    if any(a['max_target']>a['t'] for a in logs):raise ValueError('future label')
for row in r['scores']:
    vals=[]
    for yr in ['2020','2021']:
        mask=z['years']==yr
        for h in [1,5,20]:vals.append(np.mean([np.mean((z[f"{row['kind']}_{s}"][mask,h-1]-z['truth'][mask,h-1])**2) for s in [17,29]])/np.mean(z['truth'][mask,h-1]**2))
    if not np.isfinite(vals).all() or abs(np.mean(vals)-row['score'])>1e-12:raise ValueError('score')
if len(r['causal'])!=2 or not all(v['passed'] for v in r['causal']):raise ValueError('causal')
print(json.dumps(dict(verified=True,origins=451,rollouts=4,updates_each=451,labels_each=9020,scores={v['kind']:v['score'] for v in r['scores']}),indent=2))
