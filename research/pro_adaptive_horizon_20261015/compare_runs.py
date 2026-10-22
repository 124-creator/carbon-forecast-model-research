"""Actual cross-environment comparison, never a substitute for scientific validation."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def jload(p):return json.loads(Path(p).read_text())

def write(p,o):
    p=Path(p)
    if p.exists():raise FileExistsError('Keep previous comparison')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def compare(local,remote,destination):
    a=Path(local);b=Path(remote);dest=Path(destination)
    manifests={}
    for tag,r in [('local',a),('remote',b)]:
        m=jload(r/'RESULT_MANIFEST.json');errs=[]
        for entry in m['files']:
            p=r/entry['path']
            if not p.is_file():errs.append({'path':entry['path'],'error':'missing'});continue
            raw=p.read_bytes()
            if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
                errs.append({'path':entry['path'],'error':'digest/size'})
        manifests[tag]=dict(entries=len(m['files']),hash_errors=errs)
    na={str(p.relative_to(a)) for p in a.rglob('*.npz')};nb={str(p.relative_to(b)) for p in b.rglob('*.npz')}
    arrays=0;different=0;max_difference=0.;nan_masks_equal=True;details=[];mismatched_keys=[]
    for file in sorted(na & nb):
        with np.load(a/file,allow_pickle=False) as xa,np.load(b/file,allow_pickle=False) as xb:
            if set(xa.files)!=set(xb.files):mismatched_keys.append(file)
            for name in sorted(set(xa.files)&set(xb.files)):
                v=xa[name];w=xb[name];arrays+=1
                if v.shape!=w.shape:equal=False;delta=None;mask=False
                elif v.dtype.kind in 'fc' or w.dtype.kind in 'fc':
                    mask=bool(np.array_equal(np.isnan(v),np.isnan(w)));nan_masks_equal &= mask
                    equal=bool(np.array_equal(v,w,equal_nan=True))
                    finite=np.isfinite(v)&np.isfinite(w)
                    delta=float(np.max(np.abs(v[finite]-w[finite]))) if np.any(finite) else 0.
                else:
                    equal=bool(np.array_equal(v,w));mask=True
                    delta=float(np.max(np.abs(v.astype(float)-w.astype(float)))) if v.dtype.kind in 'biu' and v.size else None
                if not equal:
                    different+=1;details.append(dict(file=file,array=name,shape_local=list(v.shape),shape_remote=list(w.shape),max_abs=delta,nan_masks_equal=mask))
                if delta is not None:max_difference=max(max_difference,delta)
    score_fields=['mean_seed_relative_MSE','ensemble_relative_MSE']
    def scores(r):
        with (r/'all_scores.csv').open() as f:return {x['model']:x for x in csv.DictReader(f)}
    sa=scores(a);sb=scores(b);score_diff={k:{f:abs(float(sa[k][f])-float(sb[k][f])) for f in score_fields} for k in sa.keys()&sb.keys()}
    scoremax=max(v for d in score_diff.values() for v in d.values())
    trace=[];checkpoints=[];tracemax={k:0. for k in ['epoch','training_MSE','penalty','loss','validation_MSE','unclipped_gradient_norm']};trace_identical=0
    fitfiles=sorted((a/'fits').glob('*.json'))
    for p in fitfiles:
        if not (b/'fits'/p.name).exists():raise FileNotFoundError(p.name)
        u=jload(p);v=jload(b/'fits'/p.name);x=np.array(u['trace']);y=np.array(v['trace']);n=min(len(x),len(y))
        maxima=np.max(np.abs(x[:n]-y[:n]),axis=0) if n else np.zeros(6)
        exact=bool(x.shape==y.shape and np.array_equal(x,y))
        if exact:trace_identical+=1
        else:trace.append(dict(file=p.name,epochs_local=len(x),epochs_remote=len(y),common_prefix_field_max_abs=maxima.tolist()))
        for k,d in zip(tracemax,maxima):tracemax[k]=max(tracemax[k],float(d))
        if u['selected_epoch']!=v['selected_epoch']:checkpoints.append(dict(file=p.name,local=u['selected_epoch'],remote=v['selected_epoch']))
    ra=jload(a/'RESULTS.json');rb=jload(b/'RESULTS.json')
    source_same=jload(a/'EXECUTED_SOURCE_HASHES.json')==jload(b/'EXECUTED_SOURCE_HASHES.json')
    data_same=jload(a/'DATA_RECEIPT.json')==jload(b/'DATA_RECEIPT.json')
    exact=(na==nb and not mismatched_keys and different==0 and not trace and source_same and data_same)
    summary=dict(remote_run=jload(b/'REMOTE_EXECUTION.json') if (b/'REMOTE_EXECUTION.json').exists() else None,
        strict_numeric_reproduction_passed=exact,source_hashes_identical=source_same,source_count=len(jload(a/'EXECUTED_SOURCE_HASHES.json')),data_prefix_identical=data_same,
        manifests=manifests,npz_files_compared=len(na&nb),local_only_npz=sorted(na-nb),remote_only_npz=sorted(nb-na),mismatched_archive_keys=mismatched_keys,
        arrays_compared=arrays,nonidentical_arrays=different,maximum_array_absolute_difference=max_difference,nan_masks_equal=nan_masks_equal,
        score_configurations=len(score_diff),scores_compared=len(score_diff)*len(score_fields),maximum_score_absolute_difference=scoremax,
        traces_compared=len(fitfiles),identical_traces=trace_identical,nonidentical_traces=len(trace),trace_max_abs_fields=tracemax,checkpoint_changes=checkpoints,
        scientific_status_local=ra['status'],scientific_status_remote=rb['status'],candidate_local=ra['candidate'],candidate_remote=rb['candidate'],
        environment_local=jload(a/'ENVIRONMENT.json'),environment_remote=jload(b/'ENVIRONMENT.json'),
        interpretation='Actual complete replay comparison. Execution success, strict numerical equality, scientific gate and independent holdout confirmation are separate. Preserve the first run; no favorable replacement. Environment differences are recorded, not asserted to be the isolated cause.')
    full=dict(summary=summary,array_differences=details,score_differences=score_diff,trace_differences=trace)
    write(dest,summary);write(dest.with_name(dest.stem+'_DETAILS.json'),full)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['candidate_local','candidate_remote','environment_local','environment_remote','manifests']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--local',required=True);p.add_argument('--remote',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();compare(a.local,a.remote,a.destination)
