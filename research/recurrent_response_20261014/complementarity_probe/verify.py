from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False);rs=json.loads((R/'RESULTS.json').read_text());orig=z['origins'];truth=z['truth'];years=z['years']
if len(orig)!=451:raise ValueError('count')
for row in json.loads((R/'WEIGHTS.json').read_text()):
    if row['max_target'] is not None and row['max_target']>=row['issuance']:raise ValueError('maturity')
    if not 0<=row['w']<=1:raise ValueError('convex weight')
for row in rs:
    kind=row['kind'];cells=[]
    for year in ['2020','2021']:
        mask=years==year
        for h in [1,5,20]:cells.append(np.mean([np.mean((z[f'{kind}_{seed}'][mask,h-1]-truth[mask,h-1])**2) for seed in [17,29]])/np.mean(truth[mask,h-1]**2))
    if abs(np.mean(cells)-row['score'])>1e-12:raise ValueError('score')

def replay(seed,target):
    s=z[f'slow_{seed}'];f=z[f'fast_{seed}'];out=s.copy()
    for i,t in enumerate(orig):
        for j in range(20):
            ids=np.where(orig+j+1<t)[0][-480:];w=.5
            if len(ids)>=20:
                d=f[ids,j]-s[ids,j];den=float(d@d)
                if den>1e-12:w=float(np.clip(((d@(target[ids,j]-s[ids,j]))/den+.5)/2,0,1))
            out[i,j]=s[i,j]+w*(f[i,j]-s[i,j])
    return out
for seed in [17,29]:
    a=replay(seed,truth)
    if not np.array_equal(a,z[f'delayed_weight_{seed}']):raise ValueError('weight replay')
    q=truth.copy();q[orig[:,None]+np.arange(1,21)[None]>=600]+=10000
    b=replay(seed,q)
    if not np.array_equal(a[orig<600],b[orig<600]):raise ValueError('future labels')
print(json.dumps({'verified':True,'origins':451,'weight_future_replays':2,'scores':{r['kind']:r['score'] for r in rs}}))
