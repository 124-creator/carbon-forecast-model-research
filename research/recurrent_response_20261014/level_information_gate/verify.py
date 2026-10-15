import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());logs=json.loads((R/'FIT_LOGS.json').read_text())
want={(m,s,b) for m in ['zero','level'] for s in [17,29] for b in range(12)}
if len(logs)!=48 or {(v['mode'],v['seed'],v['block']) for v in logs}!=want or r['fits']!=48:raise ValueError('fit identities')
for v in logs:
 if v['end']!=459+40*v['block'] or v['max_training_target']>=v['validation_first'] or v['max_validation_target']>=v['end'] or v['scale_max_source']>=v['validation_first']:raise ValueError('maturity and purge')
 if not 0<=v['selected']<=v['epochs']<=200 or len(v['history'])!=v['epochs'] or v['n']>480:raise ValueError('training')
 if not np.isfinite(v['history']).all() or v['parameters']!=1300:raise ValueError('finite count')
with np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False) as z:
 orig=z['origins'];y=z['truth'];yr=z['years']
 if len(orig)!=451 or not np.all(np.diff(orig)>0) or sum(yr=='2020')!=224 or sum(yr=='2021')!=227 or y.shape!=(451,20):raise ValueError('origin scope')
 scores={}
 for mode in ['zero','level']:
  cells=[];ss=[]
  for seed in [17,29]:
   pred=z[f'{mode}_{seed}']
   if pred.shape!=y.shape or not np.isfinite(pred).all():raise ValueError('prediction')
   val=float(np.mean([np.mean((pred[yr==year,h-1]-y[yr==year,h-1])**2)/np.mean(y[yr==year,h-1]**2) for year in ['2020','2021'] for h in [1,5,20]]));ss.append(val)
   if abs(val-r['seed_scores'][mode][str(seed)])>1e-12:raise ValueError('seed score')
  for year in ['2020','2021']:
   mask=yr==year
   for h in [1,5,20]:
    val=float(np.mean([np.mean((z[f'{mode}_{seed}'][mask,h-1]-y[mask,h-1])**2) for seed in [17,29]])/np.mean(y[mask,h-1]**2));cells.append(val)
    old=next(v['ratio'] for v in r['scores'][mode]['cells'] if v['year']==year and v['h']==h)
    if abs(val-old)>1e-12:raise ValueError('cell score')
  scores[mode]=float(np.mean(cells))
  if abs(scores[mode]-np.mean(ss))>1e-12 or abs(scores[mode]-r['scores'][mode]['score'])>1e-12:raise ValueError('aggregate score')
 gate=scores['level']<scores['zero'] and all(np.mean([v['ratio'] for v in r['scores']['level']['cells'] if v['year']==year])<np.mean([v['ratio'] for v in r['scores']['zero']['cells'] if v['year']==year]) for year in ['2020','2021'])
 if gate!=r['development_gate']:raise ValueError('gate')
if {v['mode'] for v in r['causal']}!={'zero','level'} or len(r['causal'])!=2 or not all(v['passed'] for v in r['causal']):raise ValueError('future replay receipts')
if not r['fixtures']['passed'] or r['audit']['records']!=950 or r['audit']['boundary_requested']:raise ValueError('audit')
print(json.dumps(dict(verified=True,fits=48,origins=451,scores=scores,development_gate=gate,scope='independent numeric scorer, not fresh training or independent acquisition audit'),indent=2))
