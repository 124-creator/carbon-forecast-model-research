import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('joint',ROOT/'run.py')
if spec is None or spec.loader is None: raise RuntimeError('source missing')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rows=json.loads((ROOT/'results.json').read_text()); audits=json.loads((ROOT/'actual_finite_step.json').read_text()); receipt=json.loads((ROOT/'receipt.json').read_text())
if len(rows)!=10 or len(audits)!=4800 or not receipt['complete']: raise RuntimeError('incomplete')
errors=[]; summary={}; diagnostic={}
for row in rows:
 seed=row['seed']; method=row['method']; x,y=m.pilot.data(seed)
 with np.load(ROOT/f'data_{seed}.npz',allow_pickle=False) as d:
  if not np.array_equal(d['x'],x) or not np.array_equal(d['y'],y): raise RuntimeError('data changed')
 with np.load(ROOT/f'fit_{seed}_{method}.npz',allow_pickle=False) as f:
  pred=np.array([m.pilot.forward(f['p_by_task'][t],x,'shared_private')[0][t] for t in range(4)])
  errors.append(float(np.max(np.abs(pred-f['pred']))))
  for name,sl in [('train',slice(0,64)),('validation',slice(64,128)),('test',slice(128,None))]: errors.append(float(np.max(np.abs(((pred[:,sl]-y[:,sl])**2).mean(1)-row[name+'_mse']))))
  if len(f['final_p'])!=340: raise RuntimeError('parameter count')
 summary.setdefault(method,[]).append(float(np.mean(row['test_mse'])))
for seed in [17,29]:
 x,y=m.pilot.data(seed)
 for method in ['joint_adaptation','shared_eval_ablation']:
  p=m.pilot.init(seed); records=[a for a in audits if a['seed']==seed and a['method']==method]; accepted=0; violations=0; trial_increases=0
  if len(records)!=1200: raise RuntimeError('missing finite-step records')
  for a in records:
   p,d=m.candidate(p,x[:,:64],y[:,:64],method=='shared_eval_ablation')
   if d['accepted']!=a['accepted']: raise RuntimeError('accept mismatch')
   for key in ['before','trial_after','actual_after','gate_before','gate_after']: errors.append(float(np.max(np.abs(np.array(d[key])-a[key]))))
   accepted+=a['accepted']; violations+=int(np.any(np.array(a['actual_after'])>np.array(a['before'])+1e-10)); trial_increases+=int(np.any(np.array(a['trial_after'])>np.array(a['before'])+1e-10))
  with np.load(ROOT/f'fit_{seed}_{method}.npz',allow_pickle=False) as f: errors.append(float(np.max(np.abs(p-f['final_p']))))
  diagnostic[f'{seed}_{method}']=dict(accepted=accepted,steps=1200,actual_vs_private_only_increases=violations,trial_vs_private_only_increases=trial_increases)
if max(errors)>1e-10: raise RuntimeError(('replay mismatch',max(errors)))
result=dict(verified_runs=len(rows),verified_actual_steps=len(audits),max_replay_error=max(errors),seconds=receipt['seconds'],test_mse_by_seed=summary,test_mse_mean={k:float(np.mean(v)) for k,v in summary.items()},finite_step=diagnostic,source_sha256=hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest())
(ROOT/('VERIFY_OPTIMIZED.json' if not __debug__ else 'VERIFY.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
