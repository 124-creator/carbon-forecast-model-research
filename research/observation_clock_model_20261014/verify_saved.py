"""Independent saved-prediction scoring; no training or pickle loading."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
rows=json.loads((ROOT/'RESULTS.json').read_text())
out=[]
seen=set()
for r in rows:
    key=(r['regime'],r['seed'],r['model'])
    if key in seen: raise RuntimeError('duplicate configuration')
    seen.add(key)
    with np.load(ROOT/f'{key[0]}_{key[1]}_{key[2]}.npz',allow_pickle=False) as z:
        a=z['prediction']; y=z['target']; m=z['true_mean']; q=z['q']; k=z['future_k_diagnostic_only']
        if a.shape!=(128,3) or not np.isfinite(a).all(): raise RuntimeError('bad predictions')
        mse=float(np.mean(np.square(a-y)))
        ce=float(-np.mean(np.log(q[np.arange(128)[:,None],np.arange(3),k]+1e-12)))
        if abs(mse-r['mse'])>1e-12 or abs(ce-r['q_ce'])>1e-12: raise RuntimeError('metric mismatch')
        if not np.allclose(q.sum(2),1,atol=1e-12): raise RuntimeError('bad clock distribution')
        out.append(dict(regime=key[0],seed=key[1],model=key[2],mse=mse,
                        rw_mse=float(np.mean(y*y)),bayes_mean_mse=float(np.mean((m-y)**2)),
                        q_ce=ce,mean_recovery_mse=float(np.mean((a-m)**2))))
if len(rows)!=30: raise RuntimeError(f'incomplete fixed run: {len(rows)} / 30')
# Identical auxiliaries are intentionally algebraically redundant; disclose, don't count twice.
for regime in ['unknown_state','no_information','known_calendar']:
    for seed in [7,17]:
        with np.load(ROOT/f'{regime}_{seed}_auxiliary.npz',allow_pickle=False) as a, np.load(ROOT/f'{regime}_{seed}_no_condition.npz',allow_pickle=False) as b:
            if not np.allclose(a['prediction'],b['prediction'],atol=1e-12): raise RuntimeError('redundant control unexpectedly differs')
receipt={'count':len(rows),'all_metrics_replayed':True,'auxiliary_no_condition_identical':True,
         'optimized':not __debug__,'results':out}
name='VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json'
(ROOT/name).write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='results'},indent=2))
for regime in ['unknown_state','no_information','known_calendar']:
    print(regime,{mode:float(np.mean([r['mse'] for r in out if r['regime']==regime and r['model']==mode])) for mode in ['original','gap','auxiliary','no_condition','marginal']})
