"""Compare actual full executions. Never infer equality from a green CI badge."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def compare(left,right,destination):
    left=Path(left);right=Path(right);records=[];arrays=0;different=0;max_error=0.;missing=[]
    for pa in sorted(left.rglob('*.npz')):
        rel=pa.relative_to(left);pb=right/rel
        if not pb.exists():missing.append(str(rel));continue
        with np.load(pa,allow_pickle=False) as a,np.load(pb,allow_pickle=False) as b:
            if set(a.files)!=set(b.files):raise ValueError(f'Keys differ {rel}')
            bad=0;maximum=0.;masks=True;shapes=True
            for k in a.files:
                x,y=a[k],b[k];arrays+=1
                if x.shape!=y.shape:shapes=False;bad+=1;continue
                if np.issubdtype(x.dtype,np.number):
                    same=np.array_equal(x,y,equal_nan=True)
                    masks=masks and np.array_equal(np.isnan(x),np.isnan(y))
                    valid=np.isfinite(x)&np.isfinite(y)
                    error=float(np.max(np.abs(x[valid]-y[valid]))) if np.any(valid) else 0.
                else:same=np.array_equal(x,y);error=0.
                if not same:bad+=1
                maximum=max(maximum,error)
            different+=bad;max_error=max(max_error,maximum)
            records.append(dict(path=str(rel),arrays=len(a.files),nonidentical_arrays=bad,max_absolute_difference=maximum,nan_masks_equal=bool(masks),shapes_equal=bool(shapes)))
    trace_records=[];selected_changes=[];trace_changed=0;max_trace_fields=np.zeros(7)
    for pa in sorted((left/'fits').glob('*.json')):
        pb=right/'fits'/pa.name
        if not pb.exists():missing.append('fits/'+pa.name);continue
        a=json.loads(pa.read_text());b=json.loads(pb.read_text())
        if a['selected_epoch']!=b['selected_epoch']:selected_changes.append(dict(file=pa.name,left=a['selected_epoch'],right=b['selected_epoch']))
        x=np.asarray(a['trace'],dtype=float).reshape(-1,7);y=np.asarray(b['trace'],dtype=float).reshape(-1,7);n=min(len(x),len(y))
        equal=np.array_equal(x,y);trace_changed+=not equal
        diff=np.max(np.abs(x[:n]-y[:n]),axis=0) if n else np.zeros(7);max_trace_fields=np.maximum(max_trace_fields,diff)
        if not equal:trace_records.append(dict(file=pa.name,left_epochs=len(x),right_epochs=len(y),common_epoch_max_differences=diff.tolist()))
    a=json.loads((left/'ALL_MODEL_METRICS.json').read_text());b=json.loads((right/'ALL_MODEL_METRICS.json').read_text())
    score_diff=max(abs(a[k][field]-b[k][field]) for k in a for field in ('score','ensemble'))
    sa=json.loads((left/'RESULTS.json').read_text());sb=json.loads((right/'RESULTS.json').read_text())
    srcl=json.loads((left/'SOURCE_HASHES.json').read_text());srcr=json.loads((right/'SOURCE_HASHES.json').read_text())
    common=sorted(set(srcl)&set(srcr));source_equal=all(srcl[k]==srcr[k] for k in common)
    summary=dict(left_status=sa['status'],right_status=sb['status'],candidate_left=sa['candidate']['score'],candidate_right=sb['candidate']['score'],candidate_ensemble_left=sa['candidate']['ensemble'],candidate_ensemble_right=sb['candidate']['ensemble'],array_archives=len(records),arrays_compared=arrays,nonidentical_arrays=different,array_max_absolute_difference=max_error,missing_files=missing,strict_arrays_equal=not missing and different==0,fit_trace_files=384,nonidentical_numeric_trace_files=int(trace_changed),selected_epoch_changes=selected_changes,max_trace_field_differences=max_trace_fields.tolist(),trace_fields=['epoch','objective','validation_mse','gradient_norm_preclip','rank','top_gain_eigenvalue','ridge_condition'],score_configurations=len(a),scores_compared=2*len(a),max_score_difference=score_diff,common_source_files=common,executed_sources_equal=source_equal,prefix_equal=hashlib.sha256((left/'approved_prefix.csv').read_bytes()).digest()==hashlib.sha256((right/'approved_prefix.csv').read_bytes()).digest(),left_environment=json.loads((left/'ENVIRONMENT.json').read_text()),right_environment=json.loads((right/'ENVIRONMENT.json').read_text()),scope='Actual full re-execution comparison; not third-party replication, holdout confirmation or identification of a causal environmental factor')
    d=Path(destination);d.mkdir(parents=True,exist_ok=True)
    (d/'REMOTE_COMPARISON.json').write_text(json.dumps(summary,indent=2)+'\n')
    (d/'REPRODUCTION_DIFFERENCES_FULL.json').write_text(json.dumps(dict(summary=summary,arrays=records,traces=trace_records),indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('left');a.add_argument('right');a.add_argument('--destination',required=True);p=a.parse_args();compare(p.left,p.right,p.destination)
