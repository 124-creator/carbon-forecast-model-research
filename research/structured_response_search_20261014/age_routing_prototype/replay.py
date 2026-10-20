# Read-only checkpoint replay and missing-prediction recovery; NEVER trains.
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from core import Route,Small,gaussian_predict
from run import dataset,masks,generate,TRUE_K
ROOT=Path(__file__).parent
rows=json.loads((ROOT/'RESULTS.json').read_text(encoding='utf-8'));checks=[];gate=[]
for fam in ['G0','G1','G2','G5']:
  for seed in [7,17]:
    chosen=[r for r in rows if r['family']==fam and r['seed']==seed]
    if not chosen:continue
    y,truth=generate(fam,seed);scale=float(np.std(np.diff(y[:1200])));cap=float(np.quantile(np.abs(np.diff(y[:1200]))/scale,.995));o,d,f,t,base=dataset(y,scale,cap);ms=masks(o,base)
    data=np.load(ROOT/f'{fam}_{seed}_synthetic.npz')
    if not np.array_equal(y,data['y']):raise RuntimeError('generator changed')
    recovered={}
    for row in chosen:
      mode=row['method'];tag=f'{fam}_{seed}_{mode}'
      if mode=='rw':p=torch.zeros_like(t)
      elif mode=='linear_oracle_DIAGNOSTIC':p=gaussian_predict(d,torch.tensor(TRUE_K),torch.tensor(truth['variances'])/scale**2)
      elif mode=='linear_oracle120_DIAGNOSTIC':
        z=np.load(ROOT/f'{fam}_{seed}_oracle120.npz');values=[float(np.mean((z['pred'][:,j][z['mask'][:,j]]-z['targets'][:,j][z['mask'][:,j]])**2)) for j in range(3)]
        if not np.allclose(values,row['test_h'],atol=1e-12):raise RuntimeError('oracle120 metrics')
        checks.append({'tag':tag,'verified':True});continue
      else:
        ck=torch.load(ROOT/(tag+'.pt'),weights_only=True)
        if mode=='mle':
          model=Route('constant');model.v.data.copy_(ck['selected'][0]);p=gaussian_predict(d,model.kernel(),ck['selected'][1].exp())
        else:
          model=Small() if mode=='small' else Route(mode);model.load_state_dict(ck['selected']);p=model(d,f)
        trace=json.loads((ROOT/(tag+'_trace.json')).read_text(encoding='utf-8'))
        if len(trace)!=row['epochs']:raise RuntimeError('trace epoch count')
        if mode!='mle':
          # Fresh gradient norm at last, not selected, parameter vector.
          model.load_state_dict(ck['last']);model.zero_grad();train=ms['train'].any(1)
          from run import loss
          l=loss(model(d[train],f[train]),t[train],ms['train'][train]);l.backward();grad=float(torch.sqrt(sum((x.grad*x.grad).sum() for x in model.parameters())))
          if not np.isclose(grad,row['last_gradient'],rtol=1e-10):raise RuntimeError('last gradient mismatch')
      p=p.detach().numpy();recovered[mode]=p
      values=[float(np.mean((p[:,j][ms['test'][:,j].numpy()]-t[:,j].numpy()[ms['test'][:,j].numpy()])**2)) for j in range(3)]
      if not np.allclose(values,row['test_h'],atol=1e-12,rtol=1e-12):raise RuntimeError(f'metric mismatch {tag}')
      checks.append({'tag':tag,'verified':True,'mean_test':float(np.mean(values))})
    np.savez_compressed(ROOT/f'{fam}_{seed}_REPLAY_predictions.npz',**recovered)
for seed in [7,17]:
  available={(r['family'],r['method']):float(np.mean(r['test_h'])) for r in rows if r['seed']==seed}
  x={'seed':seed}
  if ('G0','full') in available:x['null_full_rw_ratio']=available['G0','full']/available['G0','rw']
  if ('G1','linear_oracle_DIAGNOSTIC') in available:x['linear_full_oracle_ratio']=available['G1','full']/available['G1','linear_oracle_DIAGNOSTIC']
  if all(('G2',m) in available for m in ['full','age','evidence','constant','mle','small','rw']):
    full=available['G2','full'];x['age_increment']=1-full/available['G2','evidence'];x['evidence_increment']=1-full/available['G2','age'];x['strongest_increment']=1-full/min(available['G2',m] for m in ['age','evidence','constant','mle','small','rw'])
  else:x['age_evidence_independent_increment']='NOT_TESTED_TO_COMPLETION'
  gate.append(x)
expected=[f'{fam}_{seed}_{mode}' for fam in ['G0','G1','G2','G5'] for seed in [7,17] for mode in ['full','constant','age','evidence','small','mle']]
observed={f"{r['family']}_{r['seed']}_{r['method']}" for r in rows}
missing=[x for x in expected if x not in observed];truncated=[f"{r['family']}_{r['seed']}_{r['method']}" for r in rows if r.get('epochs',500)<(100 if r['method']=='mle' else 500)]
out={'verified_saved_rows':len(checks),'expected_train_fits':len(expected),'executed_train_fits':sum(x in observed for x in expected),'full_budget_train_fits':sum(x in observed for x in expected)-len(truncated),'missing_train_fits':missing,'truncated_train_fits':truncated,'seed_gates':gate,'engineering':json.loads((ROOT/'ENGINEERING.json').read_text()),'mechanism_claim':'UNRESOLVED' if missing else 'see complete gates','no_market_access':True,'optimization_sufficient_claim':False,'decision':'STOP current implementation budget/learning gate; research continues','budget_note':'dense float64 triangular training prevented full fixed 500epoch grid within 1200 seconds; do not conflate unrun mechanisms with negative evidence'}
(ROOT/'REPLAY_VERIFICATION.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
(ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
