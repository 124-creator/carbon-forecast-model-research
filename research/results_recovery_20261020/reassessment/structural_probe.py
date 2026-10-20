"""Read-only representational checks of archived stage-g final parameters.
No fits, new market scores, hyperparameter search, or data redistribution.
"""
from pathlib import Path
import json, hashlib
import numpy as np
BASE=Path(__file__).resolve().parent
SOURCE=BASE/'stage_g_evidence/results/FINAL_STRUCTURAL_STATES.json'
saved=json.loads(SOURCE.read_text())
def rates(a):
    s=1/(1+np.exp(-a)); low=.98*s[...,0]
    return np.stack((low,low+(.98-low)*s[...,1]),-1)
def predict(r, v, state):
    raw=np.asarray(state['raw']); rho=rates(raw)
    filters=(1-rho)[None,:]*rho[None,:]**np.arange(63,-1,-1)[:,None]
    s=r@filters
    z=np.column_stack((np.log1p(np.abs(r[:,-1])),v))/np.asarray(state['context_scale'])
    q=2*np.tanh(z@np.asarray(state['gate']).T)
    rr=rates(raw[None,:]+q)
    response=np.stack([sum(rr**j for j in range(1,h+1))*s for h in (1,5,20)],axis=1)
    return np.column_stack([1,5,20])*state['beta'][0]+response@np.asarray(state['beta'])[1:]
rng=np.random.default_rng(20261020)
r=rng.normal(0,.8,size=(256,64));v=rng.normal(size=256)
checks=[]
for cell, models in saved.items():
    state=models['anchored_duration']['model']['state']
    f=predict(r,v,state);fm=predict(-r,v,state)
    symmetry=np.max(np.abs(f+fm-2*np.asarray([1,5,20])[None]*state['beta'][0]))
    rho=rates(np.asarray(state['raw']))
    weights=(1-rho)[None,:]*rho[None,:]**np.arange(63,-1,-1)[:,None]
    constraints=np.vstack((weights.T,np.eye(64)[-1]))
    u,s,vt=np.linalg.svd(constraints,full_matrices=True)
    rank=int(np.sum(s>1e-12))
    direction=vt[rank:][0]
    direction=direction/(np.linalg.norm(direction)+1e-30)
    rr=r+direction[None]*2
    null_change=np.max(np.abs(predict(rr,v,state)-f))
    checks.append({'cell':cell,'sign_reflection_identity_max_gap':float(symmetry),
        'constraints_rank':rank,'indistinguishable_linear_subspace_dimension':64-rank,
        'paired_history_change_l2':2.,'paired_prediction_max_gap':float(null_change)})
result={'status':'PASS_READ_ONLY_STRUCTURAL_IDENTITIES','fits':0,'new_market_forecast_scores':0,
        'parameters_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'checks':checks,
        'max_sign_identity_gap':max(c['sign_reflection_identity_max_gap'] for c in checks),
        'max_nullspace_prediction_gap':max(c['paired_prediction_max_gap'] for c in checks),
        'scope':'Properties of the actual stage-g base input and final models; not a proof that asymmetry is forecastable in real data; not a statement about arbitrary EWT-augmented inputs.'}
(BASE/'STRUCTURAL_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
