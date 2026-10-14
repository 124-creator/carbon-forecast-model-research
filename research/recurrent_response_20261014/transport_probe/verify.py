import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(z['origins'])!=451 or len(r['receipts'])!=6:raise ValueError('counts')
for receipt in r['receipts']:
    if receipt['updates']!=470 or len(receipt['refreshes'])!=11:raise ValueError('budget')
    for v in receipt['refreshes']:
        if v['anchor_max']>=v['t']:raise ValueError('anchor maturity')
for row in r['scores']:
    cells=[]
    for yr in ['2020','2021']:
        mask=z['years']==yr
        for h in [1,5,20]:
            cells.append(np.mean([np.mean((z[f"{row['kind']}_{s}"][mask,h-1]-z['truth'][mask,h-1])**2) for s in [17,29]])/np.mean(z['truth'][mask,h-1]**2))
    if not np.isfinite(cells).all() or abs(np.mean(cells)-row['score'])>1e-12:raise ValueError('score')
if len(r['causal'])!=3 or not all(v['passed'] for v in r['causal']):raise ValueError('causal')
print(json.dumps(dict(verified=True,origins=451,rollouts=6,updates_each=470,scores={v['kind']:v['score'] for v in r['scores']})))
