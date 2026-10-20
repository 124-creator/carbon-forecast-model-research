"""Maturity arithmetic feasibility only; novelty gate unresolved."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
HS=[1,5,20]
CUTS=[-1.,0.,1.]

def fit(x,inc,partial,independent=False):
    ts=np.arange(60,300)
    masks=[ts+h<300 if partial else ts+20<300 for h in HS]
    candidates=[]; losses=[]
    for cut in CUTS:
        means=[]; risk=[]
        for h,mask in zip(HS,masks):
            ids=ts[mask]; y=np.array([inc[t+1:t+h+1].sum() for t in ids]); left=x[ids]<=cut
            if min(left.sum(),(~left).sum())<20:
                values=[float(y.mean())]*2
            else:values=[float(y[left].mean()),float(y[~left].mean())]
            means.append(values); pred=np.where(left,values[0],values[1]); risk.append(float(np.mean((y-pred)**2)/h))
        candidates.append(means);losses.append(risk)
    a=np.array(candidates); losses=np.array(losses)
    choices=np.argmin(losses,axis=0) if independent else np.repeat(np.argmin(losses.mean(axis=1)),3)
    return [dict(cut=CUTS[int(j)],values=a[j,k].tolist()) for k,j in enumerate(choices)]

def predict(model,x):return np.column_stack([np.where(x<=m['cut'],*m['values']) for m in model])

rows=[];checks=0
for condition in ['none','stable','shift60','shift20']:
    for seed in range(100):
        rng=np.random.default_rng(seed);x=np.zeros(650)
        for t in range(1,len(x)):x[t]=.85*x[t-1]+rng.normal()
        beta=np.ones(len(x))*.6
        if condition=='none':beta[:]=0
        if condition.startswith('shift'):beta[:300-int(condition[5:])]*=-1
        inc=rng.normal(size=len(x))+beta*np.tanh(x)
        test=np.arange(300,600);truth=np.array([[inc[t+1:t+h+1].sum() for h in HS] for t in test]); base=np.mean(truth**2,axis=0)
        results={}
        for name,partial,ind in [('complete_shared',False,False),('partial_shared',True,False),('partial_independent',True,True)]:
            model=fit(x,inc,partial,ind);q=inc.copy();q[300:]+=10000
            if model!=fit(x,q,partial,ind):raise ValueError('future label leakage')
            checks+=1
            pred=predict(model,x[test]); ratios=np.mean((truth-pred)**2,axis=0)/base
            results[name]={'ratios':ratios.tolist(),'score':float(ratios.mean())}
        rows.append({'condition':condition,'seed':seed,'results':results})
summary={c:{m:float(np.mean([r['results'][m]['score'] for r in rows if r['condition']==c])) for m in ['complete_shared','partial_shared','partial_independent']} for c in ['none','stable','shift60','shift20']}
result={'scope':'synthetic only, no novelty claim','rows':rows,'summary':summary,'future_refits':checks,'eligible_by_h':{str(h):int(np.sum(np.arange(60,300)+h<300)) for h in HS}}
(R/'RESULTS.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
