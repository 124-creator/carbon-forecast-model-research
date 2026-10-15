import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
r=json.loads((R/'RESULTS.json').read_text());logs=json.loads((R/'FIT_LOGS.json').read_text())
if len(logs)!=24 or r['fits']!=24 or {(v['seed'],v['block']) for v in logs}!={(s,b) for s in [17,29] for b in range(12)}:raise ValueError('fit identity')
for v in logs:
 if v['end']!=459+40*v['block'] or v['weight_max_target']>=v['validation_first'] or v['max_validation_target']>=v['end']:raise ValueError('purge or weight boundary')
 if not 0<=v['selected']<=v['epochs']<=200 or len(v['history'])!=v['epochs'] or v['n']>480 or v['parameters']!=1252:raise ValueError('training shape')
 w=np.array(v['weights'])
 if w.shape!=(20,) or not np.isfinite(w).all() or not np.all(w>0) or abs(w.mean()-1)>1e-12:raise ValueError('weight scale')
 for h in v['history']:
  if not np.isfinite(list(h.values())).all():raise ValueError('nonfinite')
with np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False) as z:
 y=z['truth'];yr=z['years'];orig=z['origins']
 if y.shape!=(451,20) or len(orig)!=451 or not np.all(np.diff(orig)>0) or sum(yr=='2020')!=224 or sum(yr=='2021')!=227:raise ValueError('origin scope')
 values={}
 for mode in ['plain','weighted']:
  cells=[];seed_values=[]
  for s in [17,29]:
   pred=z[f'{mode}_{s}']
   if pred.shape!=y.shape or not np.isfinite(pred).all():raise ValueError('prediction')
   score=float(np.mean([np.mean((pred[yr==year,h-1]-y[yr==year,h-1])**2)/np.mean(y[yr==year,h-1]**2) for year in ['2020','2021'] for h in [1,5,20]]));seed_values.append(score)
   if abs(score-r['seed_scores'][mode][str(s)])>1e-12:raise ValueError('seed score')
  for year in ['2020','2021']:
   mask=yr==year
   for h in [1,5,20]:
    score=float(np.mean([np.mean((z[f'{mode}_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2));cells.append(score)
    old=next(v['ratio'] for v in r['scores'][mode]['cells'] if v['year']==year and v['h']==h)
    if abs(score-old)>1e-12:raise ValueError('cell')
  values[mode]=float(np.mean(cells))
  if abs(values[mode]-r['scores'][mode]['score'])>1e-12 or abs(values[mode]-np.mean(seed_values))>1e-12:raise ValueError('aggregate')
 if abs(values['plain']-.9753725253647837)>1e-12:raise ValueError('plain historical replay')
 gate=values['weighted']<values['plain'] and all(np.mean([v['ratio'] for v in r['scores']['weighted']['cells'] if v['year']==year])<np.mean([v['ratio'] for v in r['scores']['plain']['cells'] if v['year']==year]) for year in ['2020','2021'])
 if gate!=r['development_gate']:raise ValueError('decision')
if len(r['reference_refits'])!=2 or {v['seed'] for v in r['reference_refits']}!={17,29} or not all(v['passed'] for v in r['reference_refits']):raise ValueError('plain refit receipts')
if len(r['causal'])!=1 or not r['causal'][0]['passed'] or not r['fixtures']['passed']:raise ValueError('future or fixture receipts')
print(json.dumps(dict(verified=True,fits=24,origins=451,scores=values,development_gate=gate,scope='independent numeric scorer and receipts; not independent retraining'),indent=2))
