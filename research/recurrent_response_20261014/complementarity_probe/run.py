from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
slow=np.load(R.parent/'local_only/PREDICTIONS.npz',allow_pickle=False)
fast=np.load(R.parent/'online_group_probe/local_only/PREDICTIONS.npz',allow_pickle=False)
if not np.array_equal(slow['origins'],fast['origins']) or not np.array_equal(slow['truth'],fast['truth']):raise ValueError('matching')
orig=slow['origins'];truth=slow['truth'];years=slow['years'];saved={'origins':orig,'truth':truth,'years':years};logs=[]
for seed in [17,29]:
    s=slow[f'GRUfree_{seed}'];f=fast[f'ordinary_{seed}'];saved[f'slow_{seed}']=s;saved[f'fast_{seed}']=f;saved[f'average_{seed}']=(s+f)/2
    out=s.copy()
    for i,t in enumerate(orig):
        for j in range(20):
            ids=np.where(orig+j+1<t)[0][-480:];w=.5
            if len(ids)>=20:
                d=f[ids,j]-s[ids,j];res=truth[ids,j]-s[ids,j];den=float(d@d)
                if den>1e-12:w=float(np.clip(((d@res)/den+.5)/2,0,1))
            out[i,j]=s[i,j]+w*(f[i,j]-s[i,j])
            if j in [0,4,19]:logs.append(dict(seed=seed,issuance=int(t),h=j+1,n=len(ids),max_target=int(orig[ids[-1]]+j+1) if len(ids) else None,w=w))
    saved[f'delayed_weight_{seed}']=out
result=[]
for kind in ['slow','fast','average','delayed_weight']:
    cells=[]
    for yr in ['2020','2021']:
        mask=years==yr
        for h in [1,5,20]:
            v=np.mean([np.mean((saved[f'{kind}_{seed}'][mask,h-1]-truth[mask,h-1])**2) for seed in [17,29]])/np.mean(truth[mask,h-1]**2)
            cells.append(dict(year=yr,h=h,ratio=float(v)))
    result.append(dict(kind=kind,score=float(np.mean([a['ratio'] for a in cells])),cells=cells))
local=R/'local_only';local.mkdir(exist_ok=True);np.savez_compressed(local/'PREDICTIONS.npz',**saved)
(R/'WEIGHTS.json').write_text(json.dumps(logs));(R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
