from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
y=z['truth'];years=z['years']
def score(p):
    cells=[]
    for yr in ['2020','2021']:
        mask=years==yr
        for h in [1,5,20]:cells.append(dict(year=yr,h=h,ratio=float(np.mean((p[mask,h-1]-y[mask,h-1])**2)/np.mean(y[mask,h-1]**2))))
    return dict(score=float(np.mean([v['ratio'] for v in cells])),cells=cells)
preds={}
for kind in ['slow','fast','average','delayed_weight']:
    for seed in [17,29]:preds[f'{kind}_{seed}']=z[f'{kind}_{seed}']
preds['slow_two_seed_mean']=(z['slow_17']+z['slow_29'])/2
preds['fast_two_seed_mean']=(z['fast_17']+z['fast_29'])/2
preds['cross_update_17_29']=(z['slow_17']+z['fast_29'])/2
preds['cross_update_29_17']=(z['slow_29']+z['fast_17'])/2
preds['all_four_mean']=(z['slow_17']+z['slow_29']+z['fast_17']+z['fast_29'])/4
results={k:score(v) for k,v in preds.items()}
# The averaging gain identity holds pointwise, not a causal attribution.
identities=[]
for a,b in [('slow_17','slow_29'),('fast_17','fast_29'),('slow_17','fast_17'),('slow_29','fast_29')]:
    lhs=((preds[a]-y)**2+(preds[b]-y)**2)/2-(((preds[a]+preds[b])/2-y)**2)
    rhs=(preds[a]-preds[b])**2/4
    err=float(np.max(abs(lhs-rhs)))
    if err>1e-10:raise ValueError('diversity identity')
    identities.append(dict(a=a,b=b,max_identity_error=err))
out=dict(origins=len(y),scores=results,identities=identities,scope='fixed two-seed diagnostic; no new model training, no hyperparameter search; unequal total training budgets across update types')
(R/'SEED_CONTROL_RESULTS.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v['score'] for k,v in results.items()},indent=2))
