from pathlib import Path
import runpy,json
R=Path(__file__).resolve().parent;v=runpy.run_path(str(R/'run.py'),run_name='diagnostic');np=v['np'];fit=v['fit_predict'];g=fit.__globals__;budget=json.loads((R/'budget.json').read_text());g['budget_seconds']=budget['actual_fit_seconds']
p,dates,initial,origins,audit=v['load_data']();receipts=json.loads((R/'fit_receipts.json').read_text());checks=json.loads((R/'causality.json').read_text());z=np.load(R/'predictions_LOCAL_ONLY.npz',allow_pickle=False)
t=initial+80;ts=origins[(origins>=t)&(origins<t+40)];changed=p.copy();changed[t+1:]+=10000+np.arange(len(p)-t-1)*17
for cls in [v['NHITS'],v['DLinear']]:
 for seed in [17,29]:
  ref=next(r for r in receipts if r['t']==t and r['model']==cls.__name__ and r['seed']==seed);pred,r=fit(changed,t,ts,cls,seed,f'b02_{cls.__name__}_{seed}_future_changed');original=z[f'{cls.__name__}_{seed}'][np.searchsorted(origins,ts[0])]
  check=dict(model=cls.__name__,seed=seed,t=t,full_fresh_fit_steps=r['steps'],parameter_identical=r['parameter_sha256']==ref['parameter_sha256'],issuance_prediction_identical=bool(np.array_equal(pred[0],original)),perturb_from=t+1,repeat_receipt=r)
  checks.append(check);v['savej']('causality.json',checks);budget['actual_fit_seconds']=g['budget_seconds'];budget['additional_mature480_fresh_causal_fits']=len(checks)-4;v['savej']('budget.json',budget)
  if not check['parameter_identical'] or not check['issuance_prediction_identical']:raise RuntimeError('causality')
print(json.dumps(budget));print('mature480 future-perturbed fresh fits passed')
