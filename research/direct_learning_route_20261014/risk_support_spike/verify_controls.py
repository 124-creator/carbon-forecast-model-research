import json,math
from pathlib import Path
R=Path(__file__).resolve().parent
for name in ['POSITIVE_CONTROL_RESULTS.json','PERSISTENT_CONTROL_RESULTS.json']:
    data=json.loads((R/name).read_text())
    if len(data['rows'])!=400: raise ValueError('row count')
    for signal in [0,1,4,8]:
        rows=[r for r in data['rows'] if r['signal']==signal]
        if sorted(r['seed'] for r in rows)!=list(range(100)):raise ValueError('seed coverage')
        for method in ['empirical','iid_lcb','block_lcb']:
            ratios=[r['results'][method]['mse']/r['results'][method]['rw_mse'] for r in rows]
            if not all(math.isfinite(v) for v in ratios):raise ValueError('nonfinite')
            saved=data['summary'][str(signal)][method]
            if abs(sum(ratios)/100-saved['mean_mse_ratio'])>1e-12:raise ValueError('score')
            if sum(r['results'][method]['active'] for r in rows)!=saved['active_count']:raise ValueError('count')
    print(name,'400 trials verified')
