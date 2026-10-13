"""Independent saved-score and delayed-observation analytic checks."""
import json
from pathlib import Path
import numpy as np
from probe import delay_filter
P=Path(__file__).resolve().parent
rho=.9;q=.2;r=.1;variance=q/(1-rho*rho)
a=delay_filter(np.zeros(4),[(2,0,3.,r)],rho,q,0.,1.)
expected=rho**2*variance/(variance+r)*3.
error=abs(a[2]-expected)
if error>1e-12:raise ValueError('delayed conditional Gaussian mean mismatch')
res=json.loads((P/'ALL_RESULTS.json').read_text(encoding='utf-8'));out=[];rng=np.random.default_rng(20261013)
for scenario in ['variable_delay','zero_delay','null']:
    records={(x['seed'],x['mode']):x['score'] for x in res if x['scenario']==scenario}
    for comparator in ['price_only','hold','hold_age','fixedR','naive']:
        differences=np.array([records[(s,'aware')]-records[(s,comparator)] for s in range(10)])
        means=np.mean(differences[rng.integers(0,10,size=(10000,10))],axis=1)
        out.append(dict(scenario=scenario,comparator=comparator,mean_paired_score_difference=float(differences.mean()),synthetic_seed_bootstrap_95=np.quantile(means,[.025,.975]).tolist(),aware_wins=int((differences<0).sum()),n_seeds=10,not_market_inference=True))
    for seed in range(10):
        with np.load(P/f'synthetic_{scenario}_{seed}.npz',allow_pickle=False) as f:
            if scenario=='zero_delay' and not np.array_equal(f['aware'],f['naive']):raise ValueError('zero-delay identity failed')
            if not np.array_equal(f['random_walk'],np.zeros((451,3))):raise ValueError('random walk')
obj=dict(analytic_delayed_posterior_error=float(error),zero_delay_prediction_identity=True,paired_results=out,scope='seed uncertainty under this simulator only; not independent full retraining')
(P/('MECHANISM_OPTIMIZED.json' if not __debug__ else 'MECHANISM.json')).write_text(json.dumps(obj,indent=2),encoding='utf-8')
print(json.dumps(obj,indent=2))
