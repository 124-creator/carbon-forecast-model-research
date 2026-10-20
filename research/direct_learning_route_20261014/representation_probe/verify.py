import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(z['origins'])!=451 or len(r['receipts'])!=36:raise ValueError('counts')
for a in r['receipts']:
    if a['latest_label']>=a['fit_end'] or a['n']>480:raise ValueError('maturity')
for kind in ['raw60','raw120','compact']:
    pred=z[kind];truth=z['truth']
    if not np.isfinite(pred).all():raise ValueError('nonfinite')
    cells=[]
    for year in ['2020','2021']:
        mask=z['years']==year
        cells.extend((np.mean((pred[mask]-truth[mask])**2,0)/np.mean(truth[mask]**2,0)).tolist())
    if abs(np.mean(cells)-r['results'][kind]['score'])>1e-12:raise ValueError('score')
if len(r['causal'])!=3 or not all(c['passed'] for c in r['causal']):raise ValueError('refits')
print('Verified: 36 fits, 451 origins, 3 actual future refits; scores', {k:v['score'] for k,v in r['results'].items()})
