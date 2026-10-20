import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());fit=json.loads((R/'FIT_RECEIPTS.json').read_text())
z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(fit)!=36 or len(z['origins'])!=451:raise ValueError('counts')
for kind in ['strict40','time40','strict80']:
    if not np.isfinite(z[kind]).all():raise ValueError('incomplete')
    cells=[]
    for year in ['2020','2021']:
        mask=z['years']==year
        ratios=np.mean((z[kind][mask]-z['truth'][mask])**2,0)/np.mean((z['rw'][mask]-z['truth'][mask])**2,0)
        cells.extend(ratios[[0,4,19]].tolist())
    if abs(np.mean(cells)-r['scores'][kind]['score'])>1e-12:raise ValueError('score')
    rows=[a for a in fit if a['kind']==kind]
    if sorted(a['block'] for a in rows)!=list(range(12)):raise ValueError('blocks')
    if any(a['latest_label']>=a['fit_end'] or a['train_n']>480 for a in rows):raise ValueError('maturity')
    if any(a['minimum_support']<(80 if kind=='strict80' else 40) for a in rows):raise ValueError('support')
    if kind=='time40' and any(a['minimum_boxes']<4 for a in rows):raise ValueError('time')
if hashlib.sha256((R.parent/'baseline/vendor/mbtr/mbtr.py').read_bytes()).hexdigest()!=r['source_hash']:raise ValueError('source')
if not r['causal_refits'] or not all(a['passed'] for a in r['causal_refits']):raise ValueError('causal')
print(json.dumps({'passed':True,'fits':len(fit),'origins':451,'score':{k:v['score'] for k,v in r['scores'].items()},'time_constraint_prediction_difference':float(np.max(abs(z['time40']-z['strict40']))),'saved_fit_seconds':sum(a['seconds'] for a in fit)}))
