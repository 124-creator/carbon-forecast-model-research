"""Synthetic feasibility only: chronological split, no market data."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
H=20

def trial(seed, signal):
    rng=np.random.default_rng(seed)
    x=np.zeros(1200)
    for t in range(1,len(x)): x[t]=.85*x[t-1]+rng.normal()
    increments=rng.normal(size=len(x))+.12*signal*np.tanh(x)
    y=np.array([increments[t+1:t+H+1].sum() for t in range(1000)])
    train=np.arange(60,280); val=np.arange(300,460); test=np.arange(480,980)
    # Fixed cutpoints, not chosen using validation or test.
    candidates=[]
    for cut in [-1.,0.,1.]:
        left=x[train]<=cut
        values=[float(y[train][left].mean()),float(y[train][~left].mean())]
        candidates.append(np.where(x<=cut,values[0],values[1]))
    gains=np.array([y[val]**2-(y[val]-p[val])**2 for p in candidates])
    means=gains.mean(axis=1)
    iid=gains.std(axis=1,ddof=1)/np.sqrt(len(val))
    blocks=gains.reshape(3,4,40).mean(axis=2)
    se=blocks.std(axis=1,ddof=1)/2
    outputs={}
    for name,criterion in [('empirical',means),('iid_lcb',means-1.96*iid),('block_lcb',means-1.96*se)]:
        j=int(np.argmax(criterion)); active=bool(criterion[j]>0)
        pred=candidates[j][test] if active else np.zeros(len(test))
        outputs[name]={'active':active,'mse':float(np.mean((y[test]-pred)**2)),'rw_mse':float(np.mean(y[test]**2))}
    return outputs

rows=[]
for signal in [0,1,4,8]:
    for seed in range(100): rows.append({'signal':signal,'seed':seed,'results':trial(seed,signal)})
summary={}
for signal in [0,1,4,8]:
    subset=[r for r in rows if r['signal']==signal]
    summary[str(signal)]={name:{'active_count':sum(r['results'][name]['active'] for r in subset),'mean_mse_ratio':float(np.mean([r['results'][name]['mse']/r['results'][name]['rw_mse'] for r in subset]))} for name in ['empirical','iid_lcb','block_lcb']}
result={'scope':'synthetic feasibility, not novelty or market evidence','seed_count':100,'horizon':H,'validation_blocks':4,'block_length':40,'limitations':['four blocks cannot support a normal confidence guarantee','candidate selection multiplicity uncorrected','block boundaries still share future increments','single stump is not MBTR'],'summary':summary,'rows':rows}
(R/'POSITIVE_CONTROL_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
