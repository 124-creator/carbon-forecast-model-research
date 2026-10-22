"""Compare actual local/remote artifacts; never assume matching CI means equality."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

def same_array(a,b):
    if a.shape!=b.shape:return False,None,False
    if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
        masks=np.array_equal(np.isnan(a),np.isnan(b));finite=np.isfinite(a)&np.isfinite(b)
        delta=float(np.max(abs(a[finite]-b[finite]))) if np.any(finite) else 0.
        return np.array_equal(a,b,equal_nan=True),delta,masks
    return np.array_equal(a,b),float(np.max(abs(a.astype(float)-b.astype(float)))) if a.dtype.kind in 'biu' and a.size else 0.,True

def compare(local,remote,out):
    local,remote=Path(local),Path(remote);errors=[]
    manifest=json.loads((remote/'RESULT_MANIFEST.json').read_text())['files']
    for row in manifest:
        p=(remote/row['path']).resolve()
        if not p.is_relative_to(remote.resolve()):raise ValueError('Manifest path outside artifact')
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:errors.append(row['path'])
    require=lambda x,msg: (_ for _ in ()).throw(ValueError(msg)) if not x else None
    la={str(p.relative_to(local)):p for p in local.rglob('*.npz')};ra={str(p.relative_to(remote)):p for p in remote.rglob('*.npz')}
    require(la.keys()==ra.keys(),'Different numerical archive set')
    arrays=0;nonidentical=0;maxdiff=0.;mask_same=True;details=[]
    for name in sorted(la):
        with np.load(la[name],allow_pickle=False) as a,np.load(ra[name],allow_pickle=False) as b:
            require(set(a.files)==set(b.files),'Different arrays within '+name)
            count=0;md=0.
            for k in a.files:
                eq,df,mask=same_array(a[k],b[k]);arrays+=1;nonidentical+=not eq;count+=not eq;mask_same &= mask
                if df is not None:md=max(md,df);maxdiff=max(maxdiff,df)
            if count:details.append(dict(path=name,nonidentical=count,max_abs_error=md))
    def scores(p):return {r['name']:r for r in csv.DictReader((p/'all_scores.csv').open())}
    sl,sr=scores(local),scores(remote);require(sl.keys()==sr.keys(),'Different score configurations');sd=[]
    for n in sl:
        for c in ('mean_seed_relative_MSE','prediction_ensemble_relative_MSE'):sd.append(abs(float(sl[n][c])-float(sr[n][c])))
    checkpoints=0;traces=0;trace_changed=0;trace_error=0.
    for f in sorted((local/'fits').glob('*.json')):
        a=json.loads(f.read_text());b=json.loads((remote/'fits'/f.name).read_text());checkpoints+=a['selected_epoch']!=b['selected_epoch'];traces+=1
        eq,df,_=same_array(np.array(a['trace']),np.array(b['trace']));trace_changed+=not eq
        if df is not None:trace_error=max(trace_error,df)
    h1=json.loads((local/'SOURCE_HASHES.json').read_text());h2=json.loads((remote/'SOURCE_HASHES.json').read_text())
    required=('core.py','experiment.py','baselines.py','analyze.py','audit.py','run_all.py','test_core.py')
    hs={n:h1.get(n)==h2.get(n) for n in required}
    r1=json.loads((local/'RESULTS.json').read_text());r2=json.loads((remote/'RESULTS.json').read_text())
    result=dict(archive_files=len(la),arrays_compared=arrays,nonidentical_arrays=int(nonidentical),max_array_error=maxdiff,nan_masks_equal=bool(mask_same),
      configurations=len(sl),scores_compared=len(sd),max_score_error=max(sd),fit_traces_compared=traces,nonidentical_traces=trace_changed,max_trace_error=trace_error,
      changed_checkpoint_selections=checkpoints,source_hashes_equal=hs,source_file_count=len(required),remote_manifest_entries_verified=len(manifest),remote_manifest_errors=errors,
      price_prefix_equal=hashlib.sha256((local/'approved_prefix.csv').read_bytes()).digest()==hashlib.sha256((remote/'approved_prefix.csv').read_bytes()).digest(),
      strict_numeric_replay_passed=nonidentical==0 and trace_changed==0 and max(sd)==0 and not errors and all(hs.values()),
      local_status=r1['status'],remote_status=r2['status'],local_score=r1['main_score'],remote_score=r2['main_score'],
      local_environment=json.loads((local/'ENVIRONMENT.json').read_text()),remote_environment=json.loads((remote/'ENVIRONMENT.json').read_text()),
      differing_archives=details,scope='Actual complete numerical artifact comparison. Not third-party scientific replication or independent market validation. Platform/time log bytes need not match.')
    Path(out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('local_environment','remote_environment','differing_archives')},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--local',required=True);p.add_argument('--remote',required=True);p.add_argument('--out',required=True);a=p.parse_args();compare(a.local,a.remote,a.out)
