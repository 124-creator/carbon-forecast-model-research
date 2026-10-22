"""Post-execution comparison utility; does not train, tune or alter either result set."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def run(local,remote,destination):
    a=Path(local);b=Path(remote);differences=[];total_arrays=0;changed=0;maximum=0.;shapes_ok=True
    numeric_files=0
    for rel in sorted(p.relative_to(a) for p in a.rglob('*.npz')):
        if not (b/rel).is_file():raise FileNotFoundError(str(b/rel))
        numeric_files+=1
        with np.load(a/rel,allow_pickle=False) as x,np.load(b/rel,allow_pickle=False) as y:
            if set(x.files)!=set(y.files):raise ValueError(f'Array key mismatch {rel}')
            for k in x.files:
                u=x[k];v=y[k];total_arrays+=1
                if u.shape!=v.shape:shapes_ok=False;changed+=1;differences.append(dict(file=str(rel),key=k,reason='shape'));continue
                if np.issubdtype(u.dtype,np.number) and np.issubdtype(v.dtype,np.number):
                    same=np.array_equal(u,v,equal_nan=True)
                    finite=np.isfinite(u)&np.isfinite(v)
                    delta=float(np.max(np.abs(u[finite]-v[finite]))) if finite.any() else 0.
                    nan_equal=np.array_equal(np.isnan(u),np.isnan(v));maximum=max(maximum,delta)
                else:
                    same=np.array_equal(u,v);delta=0.;nan_equal=True
                if not same:changed+=1;differences.append(dict(file=str(rel),key=k,max_absolute_difference=delta,nan_masks_equal=nan_equal))
    first=json.loads((a/'ALL_MODEL_METRICS.json').read_text());second=json.loads((b/'ALL_MODEL_METRICS.json').read_text())
    if set(first)!=set(second):raise ValueError('Model identities differ')
    score_differences=[]
    for k in first:
        for metric in ('score','ensemble'):
            score_differences.append(dict(model=k,metric=metric,absolute_difference=abs(first[k][metric]-second[k][metric])))
    traces=0;traces_changed=0;checkpoints=0;trace_max=0.;loss_max=0.;grad_max=0.;trace_shape_changes=0
    for path in sorted((a/'fits').glob('*.json')):
        x=json.loads(path.read_text());y=json.loads((b/'fits'/path.name).read_text());traces+=1
        checkpoints+=int(x['selected_epoch']!=y['selected_epoch'])
        u=np.array(x['trace']);v=np.array(y['trace'])
        if u.shape!=v.shape:trace_shape_changes+=1
        if not np.array_equal(u,v):traces_changed+=1
        n=min(len(u),len(v))
        delta=np.abs(u[:n]-v[:n])
        if delta.size:
            trace_max=max(trace_max,float(delta.max()));loss_max=max(loss_max,float(delta[:,:-1].max()));grad_max=max(grad_max,float(delta[:,-1].max()))
    ma=json.loads((b/'RESULT_MANIFEST.json').read_text());bad=[]
    for e in ma['files']:
        data=(b/e['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=e['sha256'] or len(data)!=e['bytes']:bad.append(e['path'])
    srca=json.loads((a/'EXECUTED_SOURCE_HASHES.json').read_text());srcb=json.loads((b/'EXECUTED_SOURCE_HASHES.json').read_text())
    common=['core.py','experiment.py','baselines.py','analyze.py','audit.py','run_all.py','test_core.py']
    res=dict(npz_files_compared=numeric_files,arrays_compared=total_arrays,nonidentical_arrays=changed,all_shapes_equal=shapes_ok,
      maximum_array_absolute_difference=maximum,all_numeric_arrays_exact=changed==0,
      scores_compared=len(score_differences),max_score_difference=max(x['absolute_difference'] for x in score_differences),
      epoch_trace_files=traces,nonidentical_epoch_traces=traces_changed,changed_checkpoint_selections=checkpoints,trace_length_changes=trace_shape_changes,trace_difference_scope="Shared epoch rows, including trajectories with different final lengths; unmatched rows disclosed by length count",
      maximum_trace_difference=trace_max,maximum_loss_trace_difference=loss_max,maximum_unclipped_gradient_difference=grad_max,
      common_source_count=len(common),common_sources_identical=all(srca[k]==srcb[k] for k in common),
      price_prefix_identical=(a/'approved_prefix.csv').read_bytes()==(b/'approved_prefix.csv').read_bytes(),
      manifest_entries_checked=len(ma['files']),manifest_errors=bad,
      status_local=json.loads((a/'RESULTS.json').read_text())['status'],status_remote=json.loads((b/'RESULTS.json').read_text())['status'],
      local_environment=json.loads((a/'ENVIRONMENT.json').read_text()),remote_environment=json.loads((b/'ENVIRONMENT.json').read_text()),
      scope='Actual complete rerun comparison, not third-party or unseen-market replication. Runtime/log bytes are not expected to be identical.')
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(res,indent=2)+'\n');dest.with_name('REPRODUCTION_DIFFERENCES_FULL.json').write_text(json.dumps(dict(arrays=differences,scores=score_differences),indent=2)+'\n')
    print(json.dumps(res,indent=2));return res

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--local',required=True);p.add_argument('--remote',required=True);p.add_argument('--out',required=True);args=p.parse_args();run(args.local,args.remote,args.out)
