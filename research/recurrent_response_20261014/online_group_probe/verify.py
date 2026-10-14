from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(z['origins'])!=451 or len(r['receipts'])!=6:raise ValueError('coverage')
for item in r['scores']:
    kind=item['kind'];ratios=[]
    for year in ['2020','2021']:
        mask=z['years']==year
        for h in [1,5,20]:
            vals=[np.mean((z[f'{kind}_{s}'][mask,h-1]-z['truth'][mask,h-1])**2) for s in [17,29]]
            ratios.append(float(np.mean(vals)/np.mean(z['truth'][mask,h-1]**2)))
    if abs(np.mean(ratios)-item['score'])>1e-12:raise ValueError('score')
for row in r['receipts']:
    logs=json.loads((R/f"{row['kind']}_{row['seed']}_LOG.json").read_text())
    if len(logs)!=470 or sum(a['labels'] for a in logs)!=9210:raise ValueError('budget')
    for a in logs:
        if a['labels']!=min(20,a['t']-459):raise ValueError('maturity')
        if a['pool_before']<60 and a['offdiag']!=0:raise ValueError('cold support')
        if not np.isfinite([a['loss'],a['step_norm'],a['condition']]).all():raise ValueError('finite')
if len(r['causal'])!=3 or not all(c['passed'] and c['perturbed_from']==600 for c in r['causal']):raise ValueError('online refits')
print(json.dumps({'passed':True,'origins':451,'rollouts':6,'updates_each':470,'labels_each':9210,'scores':{s['kind']:s['score'] for s in r['scores']},'actual_future_refit_pairs':3}))
