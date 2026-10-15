"""Compare already completed numerical archives; never fit or access new data."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def run(local,remote,out):
 local,remote,out=Path(local),Path(remote),Path(out)
 checked=[];arrays=0;unequal=0;maxdiff=0.;nan_equal=True
 files=sorted(local.rglob('*.npz'))
 for f in files:
  rel=f.relative_to(local);other=remote/rel
  if not other.exists():raise FileNotFoundError(other)
  with np.load(f,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
   if set(a.files)!=set(b.files):raise ValueError('array keys')
   for k in a.files:
    x,y=a[k],b[k]
    if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('shape/dtype')
    numeric=x.dtype.kind in 'biufc'
    eq=np.array_equal(x,y,equal_nan=True) if numeric else np.array_equal(x,y)
    diff=0.
    if numeric:
     mask=np.isfinite(x)&np.isfinite(y)
     nan_equal=nan_equal and np.array_equal(np.isnan(x),np.isnan(y))
     if mask.any():diff=float(np.max(np.abs(x[mask].astype(float)-y[mask].astype(float))))
    arrays+=1;unequal+=not eq;maxdiff=max(maxdiff,diff)
    checked.append({'file':str(rel),'array':k,'equal':bool(eq),'max_absolute_difference':diff})
 lr=json.loads((local/'RESULTS.json').read_text());rr=json.loads((remote/'RESULTS.json').read_text())
 delta=[abs(lr['scores'][k][m]-rr['scores'][k][m]) for k in lr['scores'] for m in ['score','ensemble']]
 register=json.loads((local/'FIT_REGISTER.json').read_text());changes=0;traces=0;trace_max=0.;epoch_counts_equal=True
 for z in register:
  key=f"{z['mode']}_s{z['seed']}_b{(z['end']-459)//40:02}.json"
  a=json.loads((local/'fits'/key).read_text());b=json.loads((remote/'fits'/key).read_text())
  changes+=a['selected_epoch']!=b['selected_epoch'];x=np.asarray(a['history']);y=np.asarray(b['history'])
  if x.shape!=y.shape:epoch_counts_equal=False;traces+=1
  elif not np.array_equal(x,y):traces+=1;trace_max=max(trace_max,float(np.max(np.abs(x-y))))
 manifest=json.loads((remote/'MANIFEST.json').read_text())['files'];errors=[]
 for z in manifest:
  f=remote/z['path']
  if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=z['sha256']:errors.append(z['path'])
 summary={'local_status':lr['status'],'remote_status':rr['status'],'npz_files':len(files),'arrays_compared':arrays,
  'nonidentical_arrays':int(unequal),'maximum_array_difference':maxdiff,'nan_masks_equal':bool(nan_equal),
  'scores_compared':len(delta),'maximum_score_difference':max(delta),'checkpoint_changes':changes,
  'nonidentical_training_traces':traces,'trace_max_absolute_difference':trace_max,'epoch_counts_equal':epoch_counts_equal,
  'strict_numeric_reproduction':bool(unequal==0 and max(delta)==0 and traces==0),
  'candidate_first_score':lr['candidate']['score'],'candidate_remote_score':rr['candidate']['score'],
  'local_environment':json.loads((local/'ENVIRONMENT.json').read_text()),'remote_environment':json.loads((remote/'ENVIRONMENT.json').read_text()),
  'source_hashes_equal':json.loads((local/'SOURCE_HASHES.json').read_text())==json.loads((remote/'SOURCE_HASHES.json').read_text()),
  'remote_manifest_entries':len(manifest),'remote_manifest_hash_errors':errors,
  'scope':'same fixed protocol executed twice; not independent holdout, novelty certification or third-party scientific replication'}
 out.write_text(json.dumps(summary,indent=2));out.with_name('ARRAY_COMPARISON_DETAIL.json').write_text(json.dumps(checked,indent=2))
 print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('local');p.add_argument('remote');p.add_argument('out');a=p.parse_args();run(a.local,a.remote,a.out)
