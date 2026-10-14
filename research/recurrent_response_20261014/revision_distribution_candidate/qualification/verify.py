from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());z=np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False)
if len(z['origins'])!=451 or len(r['fits'])!=12:raise ValueError('count')
if np.max(abs(z['truth']-z['direct']))>1e-12:raise ValueError('equivalence')
for v in r['fits']:
    if v['max_variance_target']>=v['end'] or v['max_mean_target']>=v['variance_first']:raise ValueError('maturity')
if not r['future_refit_passed'] or not np.isfinite(z['mean']).all() or not (z['variance']>0).all():raise ValueError('finite/causal')
for v in r['results']:
    mask=np.ones(451,dtype=bool) if v['year']=='all' else z['years']==v['year'];e=(z['truth'][mask]-z['mean'][mask])**2
    ratio=float(np.mean(e)/np.mean(z['truth'][mask]**2));cn=float(np.mean(.5*(np.log(2*np.pi*z['constant'][mask])+e/z['constant'][mask])));vn=float(np.mean(.5*(np.log(2*np.pi*z['variance'][mask])+e/z['variance'][mask])))
    if not np.allclose([ratio,cn,vn],[v['mean_mse_ratio'],v['constant_nll'],v['conditional_nll']],rtol=0,atol=1e-12):raise ValueError('score')
print(json.dumps(dict(verified=True,origins=451,revision_fits=12,identity_max_error=float(np.max(abs(z['truth']-z['direct']))),results=r['results']),indent=2))
