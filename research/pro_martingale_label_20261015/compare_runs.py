"""Compare two completed runs. Exactness is measured, never inferred from CI."""
import argparse,json
from pathlib import Path
import numpy as np

def compare(a,b,out):
 a=Path(a);b=Path(b);files=arrays=different=0;largest=0.;problems=[]
 for p in sorted(a.rglob('*.npz')):
  rel=p.relative_to(a);other=b/rel
  if not other.exists():problems.append({'missing':str(rel)});continue
  files+=1
  with np.load(p,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
   if set(x.files)!=set(y.files):problems.append({'keys':str(rel)});continue
   for k in x.files:
    u,v=x[k],y[k];arrays+=1
    if u.shape!=v.shape:problems.append({'shape':[str(rel),k]});different+=1;continue
    if u.dtype.kind not in 'biufc':
     if not np.array_equal(u,v):different+=1;problems.append({'non_numeric_difference':[str(rel),k]})
     continue
    exact=np.array_equal(u,v,equal_nan=True)
    if not exact:
     different+=1
     if not np.array_equal(np.isnan(u),np.isnan(v)):problems.append({'nan_mask':[str(rel),k]})
     valid=np.isfinite(u)&np.isfinite(v)
     if valid.any():largest=max(largest,float(np.max(np.abs(u[valid].astype(float)-v[valid].astype(float)))))
 x=json.loads((a/'ALL_MODEL_METRICS.json').read_text());y=json.loads((b/'ALL_MODEL_METRICS.json').read_text())
 score_error=max(abs(x[k][f]-y[k][f]) for k in x for f in ['score','ensemble'])
 traces=changed=changed_length=0;trace_error=0.;selections=0
 for p in sorted((a/'fits').glob('*.json')):
  q=b/'fits'/p.name
  if not q.exists():problems.append({'missing_log':p.name});continue
  u=json.loads(p.read_text());v=json.loads(q.read_text());traces+=1
  selections+=int(u['selected_epoch']!=v['selected_epoch'])
  xx=np.array(u['trace']);yy=np.array(v['trace'])
  if xx.shape!=yy.shape:changed_length+=1
  if xx.shape!=yy.shape or not np.array_equal(xx,yy):changed+=1
  n=min(len(xx),len(yy))
  if n:trace_error=max(trace_error,float(np.max(np.abs(xx[:n]-yy[:n]))))
 sa=json.loads((a/'RESULTS.json').read_text());sb=json.loads((b/'RESULTS.json').read_text())
 result=dict(compared_npz_files=files,compared_arrays=arrays,nonidentical_arrays=different,max_array_absolute_difference=largest,problems=problems,compared_scores=2*len(x),max_main_ensemble_score_difference=score_error,fit_traces=traces,nonidentical_traces=changed,trace_length_changes=changed_length,max_common_trace_absolute_difference=trace_error,checkpoint_changes=selections,strict_numerical_reproduction_passed=(different==0 and changed==0 and not problems and score_error==0),scientific_status_local=sa['status'],scientific_status_remote=sb['status'],local_candidate=sa['candidate']['score'],remote_candidate=sb['candidate']['score'],scope='Complete remote reexecution and explicit numeric comparison, not external peer replication or unseen-market validation. Wall-clock and environment strings intentionally differ.')
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--local',required=True);p.add_argument('--remote',required=True);p.add_argument('--out',required=True);a=p.parse_args();compare(a.local,a.remote,a.out)
