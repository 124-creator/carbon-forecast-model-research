from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());logs=json.loads((R/'FIT_LOGS.json').read_text());causal=json.loads((R/'CAUSAL.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(logs)!=120 or len(z['origins'])!=451:raise ValueError('counts')
for row in logs:
    if not row['max_training_target']<row['validation_first'] or row['max_validation_target']>=row['fit_end']:raise ValueError('purge')
    if row['n']>480:raise ValueError('window')
    if row['selected']>row['epochs'] or len(row['history'])!=row['epochs']:raise ValueError('steps')
seed_scores={}
for entry in r['scores']:
    kind=entry['kind'];seeds=[]
    for seed in [17,29]:
        pred=z[f'{kind}_{seed}'];truth=z['truth']
        if not np.isfinite(pred).all():raise ValueError('nonfinite')
        cells=[]
        for year in ['2020','2021']:
            mask=z['years']==year
            for h in [1,5,20]:cells.append(float(np.mean((pred[mask,h-1]-truth[mask,h-1])**2)/np.mean(truth[mask,h-1]**2)))
        seeds.append(float(np.mean(cells)))
    if abs(np.mean(seeds)-entry['score'])>1e-12:raise ValueError('score')
    seed_scores[kind]=seeds
if len(causal)!=5 or not all(a['passed'] for a in causal):raise ValueError('future refits')
summary={'verified':True,'fits':120,'origins':451,'future_refits':5,'seed_scores':seed_scores,'step0_count':sum(a['selected']==0 for a in logs),'max_epoch_count':sum(a['epochs']==200 for a in logs),'parameters':{k:next(a['parameters'] for a in logs if a['kind']==k) for k in seed_scores}}
print(json.dumps(summary,indent=2))
