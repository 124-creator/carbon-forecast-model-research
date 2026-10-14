import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());logs=json.loads((R/'FIT_LOGS.json').read_text())
if len(logs)!=96 or r['fits']!=96:raise ValueError('fit count')
expected={(s,b,m) for s in [17,29] for b in range(12) for m in ['raw','single','shared','freeze']}
if {(v['seed'],v['block'],v['mode']) for v in logs}!=expected:raise ValueError('fit identities')
for v in logs:
 if v['end']!=459+40*v['block'] or v['max_training_target']>=v['validation_first'] or v['max_validation_target']>=v['end'] or v['internal_early_max_target']>=v['internal_late_first']:raise ValueError('causal boundary')
 if v['epochs']!=20 or len(v['history'])!=20 or not 0<=v['selected']<=20:raise ValueError('iterations')
 for h in v['history']:
  if not np.isfinite(list(h.values())).all():raise ValueError('nonfinite log')
  a=h['raw_encoder_norm'];b=h['used_encoder_norm']
  if v['mode']=='freeze' and b!=0:raise ValueError('freeze')
  if v['mode'] in ['single','shared']:
   if v['parameter_rank']==0 and b!=0:raise ValueError('zero rank')
   if b>0 and abs(a-b)>1e-6+1e-5*a:raise ValueError('gradient norm match')
with np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False) as z:
 if len(z['origins'])!=451 or len(np.unique(z['origins']))!=451 or not np.all(np.diff(z['origins'])>0):raise ValueError('origins')
 if sum(z['years']=='2020')!=224 or sum(z['years']=='2021')!=227:raise ValueError('year counts')
 y=z['truth'];values={}
 for mode in ['initial','raw','single','shared','freeze']:
  ratios=[]
  for yr in ['2020','2021']:
   mask=z['years']==yr
   for h in [1,5,20]:
    for seed in [17,29]:
     if z[f'{mode}_{seed}'].shape!=y.shape or not np.isfinite(z[f'{mode}_{seed}']).all():raise ValueError('prediction')
    ratio=float(np.mean([np.mean((z[f'{mode}_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2));ratios.append(ratio)
    record=next(v for v in r['scores'][mode]['cells'] if v['year']==yr and v['h']==h)
    if abs(ratio-record['ratio'])>1e-12:raise ValueError('cell score')
  values[mode]=float(np.mean(ratios))
  if abs(values[mode]-r['scores'][mode]['score'])>1e-12:raise ValueError('score')
 if abs(values['initial']-0.9753725253647837)>1e-12:raise ValueError('baseline agreement')
 cs=['initial','raw','single','freeze'];gate=all(values['shared']<values[c] for c in cs) and all(np.mean([v['ratio'] for v in r['scores']['shared']['cells'] if v['year']==yr])<np.mean([v['ratio'] for v in r['scores'][c]['cells'] if v['year']==yr]) for c in cs for yr in ['2020','2021'])
 if gate!=r['development_gate']:raise ValueError('gate')
if len(r['causal'])!=4 or {v['mode'] for v in r['causal']}!={'raw','single','shared','freeze'} or not all(v['passed'] for v in r['causal']):raise ValueError('future replay receipts')
if not r['fixtures']['passed'] or r['fixtures']['jacobian_max_error']>1e-7:raise ValueError('fixtures')
print(json.dumps(dict(verified=True,fit_count=len(logs),origins=451,development_gate=r['development_gate'],scores=values,scope='independent numeric scoring and receipts; not fresh training'),indent=2))
