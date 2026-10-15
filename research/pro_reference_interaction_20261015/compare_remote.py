"""Compare two completed output directories without modifying either experiment.
Usage: python compare_remote.py LOCAL_RESULTS REMOTE_RESULTS comparison.json
This post-training utility does not fit models or select new configurations.
"""
from pathlib import Path
import csv,json,sys
import numpy as np
local=Path(sys.argv[1]);remote=Path(sys.argv[2]);output=Path(sys.argv[3])
reports={};total_arrays=0;global_max=0.0;all_exact=True
for folder in ['fits','predictions','online']:
    left={p.name:p for p in (local/folder).glob('*.npz')}
    right={p.name:p for p in (remote/folder).glob('*.npz')}
    if set(left)!=set(right):raise ValueError(f'{folder}: unmatched archives')
    mismatch=0;arrays=0;mx=0.0;mask_match=True
    for name,p in left.items():
        with np.load(p,allow_pickle=False) as a,np.load(right[name],allow_pickle=False) as b:
            if set(a.files)!=set(b.files):raise ValueError('Unmatched arrays')
            for k in a.files:
                x,y=a[k],b[k];arrays+=1
                if x.shape!=y.shape or x.dtype.kind not in 'biuf' or y.dtype.kind not in 'biuf':raise ValueError('Shape or dtype')
                if not np.array_equal(np.isnan(x),np.isnan(y)):mask_match=False
                if not np.array_equal(x,y,equal_nan=True):mismatch+=1
                sel=np.isfinite(x)&np.isfinite(y)
                diff=float(np.max(abs(x[sel].astype(float)-y[sel].astype(float)))) if np.any(sel) else 0.0
                mx=max(mx,diff)
    reports[folder]={'archive_files':len(left),'arrays':arrays,'nonidentical_arrays':mismatch,'nan_masks_equal':mask_match,'maximum_absolute_difference':mx}
    total_arrays+=arrays;global_max=max(global_max,mx);all_exact=all_exact and mismatch==0 and mask_match
A={r['name']:r for r in csv.DictReader((local/'all_scores.csv').open())}
B={r['name']:r for r in csv.DictReader((remote/'all_scores.csv').open())}
if set(A)!=set(B):raise ValueError('Configs')
score=max(abs(float(A[n][k])-float(B[n][k])) for n in A for k in ['mean_seed_relative_MSE','prediction_ensemble_relative_MSE'])
a=json.loads((local/'RESULTS.json').read_text());b=json.loads((remote/'RESULTS.json').read_text())
result={'github_run_id':38681433247,'executed_commit':'88de468765b9a119f1b176a9656833d3952f4016','source_base_commit':'c024f140e28753bcec47f82b44300787b7f35c6b','comparisons':reports,'arrays_compared':total_arrays,'all_arrays_exact':all_exact,'max_array_absolute_difference':global_max,'score_configurations':len(A),'scores_compared':len(A)*2,'max_score_difference':score,'selected_configs_identical':a['selected']==b['selected'],'scientific_status_local':a['scientific_status'],'scientific_status_remote':b['scientific_status'],'scope':'Complete fresh remote training with the same fixed data/configuration. Not external scientific replication or independent holdout.'}
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
